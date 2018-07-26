#!/usr/bin/python3

import argparse
from collections import defaultdict, namedtuple
import math
import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import (bernoulli, multinomial, dirichlet, poisson, entropy)
from scipy.special import gamma, gammaln  # gamma _function_
from scipy.stats import gamma as gammad  # gamma _distribution

import utilities as ut

ChatSettings = namedtuple('ChatSettings',
                          ['language',  # aka theta
                           'alpha_prior',  # prior for all learners
                           'no_gamma_prior',  # no dataset size prior for listeners
                           'message_length',  # message length (default 1)
                           'speaker_hyps',  # number of hypothesised componentss in listener
                           'prior_data',  # amt of data NNS learner sees prior to chat
                           'dialogue_length',  # how long to sepak (iterations)
                           'accommodation', # accommodation parameter
                           'outputstr',  # write to file w/this name
                           ])

VERBOSE = 0  # 3 prints everything, 0 nothing.

# DEBUG settings
SET_LISTENER_LANG = False
USE_B_LANG = False
ASSYM_PRIOR = None # [0.9, 0.05, 0.05]


def set_seed():
    np.random.seed(26)

class Language:
    """Multinomial language with K cats, distributed according to theta."""

    def __init__(self, theta):
        assert len(theta) > 1, "Bad theta: %s" % ut.str_listfs(theta)
        self.theta = theta
        self.K = len(theta)

    def generate(self, N):
        """Generate 1 sample of N draws from theta.
        Sample consists of an array size K with (integer) number of draws of
        each category."""
        return multinomial(N, self.theta).rvs(1)[0]


class Learner:
    """Learner is a Dir-Multinomial."""

    def __init__(self, prior, K):
        """Prior is an array of hyperparameters of the same size as the
        language. (If it's a float, is interpreted as a symmetric prior.)

        K is an int representing number of categories in the language.
        (Can also be a Language, in which case only its size is used.)
        """

        if type(K) != int:  # Passing in Language, not dim of lang.
            if type(K) is list:
                K = len(K)
            if type(K) == Language:
                K = K.K  # i.e, get language.K

        # Note that scipy.stats.dirichlet wants priors/alphas to sum to <= 1
        # (so, normalize)
        if type(prior) == float:  # symmetric prior
            prior = [prior/K] * K  # turn into proper array

        assert len(prior) == K, "Mismatch! K %d len prior %s" % (K, prior)

        self.K = K
        self.prior = prior  # prior does not get updated.
        self.alpha = np.array(prior)  # updated posterior | prior

        self.H_trajectory = []  # record language entropy at every update

    def __str__(self):
        return 'alpha:' + ut.str_listfs(self.alpha)

    def update(self, observations):
        assert len(observations) == self.K
        self.alpha = self.alpha + np.array(observations)

    def exposure(self):
        """Number of datapoints seen by learner."""
        return sum(self.alpha) - sum(self.prior)

    def mean(self):  # What is the point of this??
        return dirichlet.mean(self.alpha)

    def var(self):
        return dirichlet.var(self.alpha)

    def std(self):
        return dirichlet.std(self.alpha)

    def pr(self, observations, extra_obs=None):
        """P(obs|alpha), distribution over obs given alpha, marg. out mult.
        Extra obs occur in the context of weight updates, where we want to keep
        track of previously heard items to add to the prior (but not actually
        update/change the posterior).
        """
        n = sum(observations)

        if n == 1:
            k = np.argmax(observations)
            return self.pr_cat(k, extra_obs)

        assert len(observations) == self.K, 'K %d obs %d' % (self.K,
                                                             len(observations))
        assert len(observations) == len(self.alpha), "alph %d, obs %d" % (
            len(self.alpha), len(observations))

        if extra_obs is not None:
            this_alpha = self.alpha + np.array(extra_obs)
            assert len(this_alpha) == len(self.alpha)
        else:
            this_alpha = self.alpha
        a0 = sum(this_alpha)
        first = (math.factorial(n) * gamma(a0))/gamma(n + a0)
        prod = 1
        for k, x in enumerate(observations):
            prod *= gamma(x + this_alpha[k]) / (math.factorial(x) *
                                                gamma(this_alpha[k]))
        return first * prod

    def pr_cat(self, k, extra_obs=None):
        """Probability of 1 observation of category k: norm'd alpha."""

        if extra_obs is not None:
            pr = (self.alpha[k] + extra_obs[k]) / (
                sum(self.alpha) + sum(extra_obs))
        else:
            pr = self.alpha[k]/sum(self.alpha)
        return pr

    def prln(self, observations):  # 'pmf' in ft_binomial
        """Log/ln probability of an observation."""
        n = sum(observations)
        a0 = sum(self.alpha)
        first = math.log(math.factorial(n)) + gammaln(a0) - gammaln(n + a0)
        prod = 0
        for k, x in enumerate(observations):
            prod += gammaln(x + self.alpha[k]) - (math.log(math.factorial(x))
                                                  + gammaln(self.alpha[k]))
        return first + prod

    def pdf(self):
        """P(theta | alpha): parallel to ft_binomial."""
        return lambda x: dirichlet.pdf(x, self.alpha)

    def entropy_dir(self):
        return dirichlet.entropy(self.alpha)

    def entropy(self):
        H = 0
        for k in range(self.K):
            p = self.pr_cat(k)
            H += p * math.log(p, 2)
        return -H

    def xentropy(self, probs):
        assert len(probs) == self.K
        H = 0
        for k in range(self.K):
            p = self.pr_cat(k)
            H += probs[k] * math.log(p, 2)
        return -H

    def update_H_trajectory(self, date):
        self.H_trajectory.append((date, self.entropy()))

    def generate(self, N=1):
        """Matches Language.generate: Generate 1 sample of N draws from theta.
        Returned sample consists of an array size K with (integer) number of
        draws of each category."""

        norm_alpha = [a/sum(self.alpha) for a in self.alpha]
        return multinomial.rvs(N, norm_alpha)

    def production(self, N, extra_obs=None):
        """Alternate generation version."""
        probs = [self.pr_cat(k, extra_obs) for k in range(self.K)]
        return multinomial.rvs(N, probs)


class Listener:
    """
    Listener tracks a mixture of possible learners (representing their
    interlocutor's hypothetical/imputed language).
    Confusingly also called speaker model in the paper, ie listener's model of
    the speaker.
    """

    def __init__(self,
                 prior,
                 language,  # can also be a Learner (w/ own language)
                 S,
                 gamma_prior=(2, 1),  # assumes |D|~=2
                 D=None,
                 weightprob=None,  # default is uniform.
                 components_data=None  # specify data, don't sample.
                 ):
        """Listener draws imputed datasets with gamma prior over poisson
        distribution over |D|, NNS's actual data exposure size."""

        self.K = language.K

        if type(prior) == float:  # symmetric prior
            prior = [prior/self.K] * self.K  # turn into proper array
        self.prior = np.array(prior)
        self.S = S

        if type(gamma_prior) == float:  # only specify shape k, scale theta=1
            gamma_prior = (gamma_prior, 1)

        if weightprob is None:  # use a uniform distribution to weight mixture
            def uniform(x):
                return 1
            weightprob = uniform
        self.weightprob = weightprob

        # represent components by their updated priors (a, b)
        self.components = [Learner(self.prior, self.K)
                           for c in range(self.S)]
        self.weights = np.array([0] * self.S, dtype=np.float64)
        # track inital data (D_z) draws (could also get from self.components)
        self.data = [np.array([0] * self.K) for c in range(self.S)]

        # Track new data that learner will add to their posterior in order to
        # match their language.
        self.heard_by_learner = np.array([0] * self.K)
        # Track data heard from learner in order to update weights over
        # potential history.
        self.heard_from_learner = np.array([0] * self.K)

        # Initialise each component by generating a unique dataset
        for s in range(self.S):
            if components_data:
                assert len(components_data) == self.S
                assert len(components_data[0]) == self.K
                obs = components_data[s]
            elif D:
                obs = language.generate(D)
            else:  # Use gamma dist prior to draw a parameter for poisson
                lambda_poisson = gammad.rvs(a=gamma_prior[0],
                                            scale=gamma_prior[1], size=1)
                D_size = poisson.rvs(mu=lambda_poisson, size=1)[0]
                # use A's own (learner) language here. (If lang=learner)
                obs = language.generate(D_size)

            self.components[s].update(obs)

            self.weights[s] = weightprob(obs)
            self.data[s] = obs

        self.weights = self.weights/sum(self.weights)
        assert_almost_equal(sum(self.weights), 1.0)

        self.H_trajectory = []  # tracks Listener entropy

    def print_hist_component_sizes(self):
        """ Using alphas as proxies for # datapoints seen."""
        print('Avg. Comp Size', np.mean([sum(s.alpha) for s in
                                         self.components]))

    def print_hist_components(self):
        cdict = defaultdict(int)
        cdict_H = defaultdict(int)
        for c in self.components:
            cdict[tuple(c.alpha)] += 1
            cdict_H[tuple(c.alpha)] = c.entropy()

        # for c in sorted(cdict, key=cdict.get, reverse=True):
        wps = []
        for c in sorted(cdict):
            count = cdict[c]
            p = c[0]/sum(c)
            w = count/len(self.components)
            print(cdict[c], ':', ut.str_listfs(c), ':: H %.2f' % cdict_H[c])
            wps.append(w*p)
        # print('wp', sum(wps))
        return wps

    def pr(self, obs, add_heard=True):
        """Renamed pmf -  pr(obs|self) """
        if add_heard is not None:
            p = sum([self.weights[s]
                     * self.components[s].pr(obs,
                                             extra_obs=self.heard_by_learner)
                    for s in range(self.S)])
        else:
            p = sum([self.weights[s] * self.components[s].pr(obs)
                    for s in range(self.S)])
        return p

    def pr_cat(self, k, add_heard=True):
        """Prob of seeing one observation of a particular category."""
        if add_heard is not None:
            p = sum([self.weights[s] * self.components[s].pr_cat(
                         k, extra_obs=self.heard_by_learner)
                    for s in range(self.S)])
        else:
            p = sum([self.weights[s] * self.components[s].pr_cat(k)
                    for s in range(self.S)])
        return p

    def entropy(self, add_heard=True):
        H = 0
        for k in range(self.K):
            if add_heard is not None:
                p = self.pr_cat(k, self.heard_by_learner)
            else:
                p = self.pr_cat(k)
            H += p * math.log(p, 2)
        return -H

    def xentropy(self, probs, add_heard=True):
        """Cross entropy between Listener's language mixture and probs."""
        assert len(probs) == self.K
        H = 0
        for k in range(self.K):
            if add_heard is not None:
                p = self.pr_cat(k, self.heard_by_learner)
            else:
                p = self.pr_cat(k)
            H += probs[k] * math.log(p, 2)
        return -H

    def macroentropy(self):
        H = np.mean([c.entropy() for c in self.components])
        return H

    def update(self, obs):
        for s, c in enumerate(self.components):
            # Weight gets updated by the likelihood/pmf/p(obs|component);
            # since the components don't get updated with new data we have to
            # add the previously heard data to the current observation.
            pr_obs = c.pr(obs, self.heard_from_learner)
            self.weights[s] = self.weights[s] * pr_obs

            # Component counts do not get updated, because these are data
            # _produced_by learner-interloc, not heard by interloc.
            # c.update(k, N)  # don't do this.
        self.weights = self.weights/sum(self.weights)  # normalise
        self.heard_from_learner += obs

    def update_heard_by_learner(self, obs):
        self.heard_by_learner += obs

    def update_H_trajectory(self, date):
        self.H_trajectory.append((date, self.entropy()))

    def production(self, N, accommodation=0.5,
                   ownlang=None):
        """
        Generate a string of length N.
        Accommodation param governs the probability of using own learned dist
        vs. infered B's language.

        Production samples first a component, then samples from that comp.
        Generate function below samples from full mixture.
        """
        if ownlang is None:  # create a learner
            ownlang = Learner(self.a0, self.b0)
            ownlang.update(self.language.generate(1000))

        acc = bernoulli.rvs(accommodation)
        if acc:
            # pick component to generate from (weighted)
            c = np.argmax(multinomial.rvs(1, self.weights))
            output = self.components[c].production(N, self.heard_by_learner)
        else:
            output = ownlang.production(N)
        return output

    def generate(self, accommodation, ownlang):
        """
        Generate one draw from full mixture of components (N=1).
        """
        acc = bernoulli.rvs(accommodation)
        if acc:
            probs = np.array([0.0] * self.K)

            for ic, c in enumerate(self.components):
                w = self.weights[ic]
                for k in range(self.K):
                    pk = c.pr_cat(k, extra_obs=self.heard_by_learner)
                    probs[k] += (w*pk)
            probs = probs/sum(probs)
            msg = multinomial.rvs(1, probs)
        else:
            msg = ownlang.generate(1)
        return msg


class Chat:

    def __init__(self,
                 chat_settings,
                 listener=True,
                 BLearner=False
                 ):
        self.s = chat_settings

        print('Chat settings:', self.s)

        self.lang = Language(self.s.language)
        self.K = len(self.s.language)

        # Always a new NS interlocutor
        self.ALearner = Learner(self.s.alpha_prior, self.K)
        self.ALearner.update(self.lang.generate(1000))
        self.ALearner.update_H_trajectory(-1)

        # Maybe an experienced NNS (in sequence of chats)
        if BLearner:
            self.BLearner = BLearner
            # null update is necessary to match AListener trajectory.
            self.BLearner.update_H_trajectory(-1)
        # Otherwise creat a new NNS
        else:
            self.BLearner = Learner(self.s.alpha_prior, self.K)
            self.BLearner.update(self.lang.generate(self.s.prior_data))
            self.BLearner.update_H_trajectory(-1)

        if listener:
            if USE_B_LANG:  # Debug/testing
                print('DEBUG: Listener lang from B')
                # AListener uses ALearner language to draw data
                gamma_prior = (max(self.BLearner.exposure(), 1.0), 1.0)
                self.AListener = Listener(self.s.alpha_prior,
                                          self.BLearner,  # XXX
                                          self.s.speaker_hyps,
                                          gamma_prior=gamma_prior)

            elif SET_LISTENER_LANG:
                print('DEBUG: setting Listener data')
                # AListener uses ALearner language to draw data
                self.AListener = Listener(self.s.alpha_prior,
                                          self.ALearner,
                                          self.s.speaker_hyps,
                                          components_data=[[1,0,0]*self.s.speaker_hyps],
                                          )
            elif self.s.no_gamma_prior:
                # All listener datasets have length=D, from true lang
                print('NO gamma')
                self.AListener = Listener(self.s.alpha_prior,
                                          self.lang,
                                          self.s.speaker_hyps,
                                          D=self.s.prior_data)
            else:
                # AListener uses ALearner language to draw data;
                # listener dataset sizes are correct for BLearner
                gamma_prior = (max(self.BLearner.exposure(), 1.0), 1.0)
                print("Setting gamma to:", gamma_prior)
                self.AListener = Listener(self.s.alpha_prior,
                                          self.ALearner,
                                          self.s.speaker_hyps,
                                          gamma_prior=gamma_prior)
            self.AListener.update_H_trajectory(-1)
        else:
            self.AListener = None

    def update_Hs(self, date):
        self.ALearner.update_H_trajectory(date)
        self.BLearner.update_H_trajectory(date)
        if self.AListener:
            self.AListener.update_H_trajectory(date)

    def chat(self, ps=False, dateshift=0):
        """Chat between A and B.
        ps collects and outputs B's probabilities/language at each timestep.
        dateshift is for iterating multiple chats with same NNS/B, different A.
        """
        if VERBOSE > 0:
            print("ACC PARAM", self.s.accommodation)
            print('INIT AList: H %.4f  B: H %.4f' % (self.AListener.entropy(),
                                                     self.BLearner.entropy()),
                  'INIT A: p0 %.3f hbyB: (%s)' % (
                      self.AListener.pr_cat(0),
                      ut.prlints(self.AListener.heard_by_learner)))
            print("Learner Alphas A %s B %s" %
                  (ut.str_listfs(self.ALearner.alpha),
                   ut.str_listfs(self.BLearner.alpha)))
            self.AListener.print_hist_components()
            print('INIT B: H %.4f (%s)' % (self.BLearner.entropy(),
                                           self.BLearner))

        if ps:
            Bps = [self.BLearner.pr_cat(1)]

        # At each dialogue step, B then A speaks (+updates).
        for date in range(self.s.dialogue_length):
            if VERBOSE > 2:
                print('step', date, '+shift', date + dateshift)

            Bmsg = self.BLearner.production(self.s.message_length)
            self.ALearner.update(Bmsg)
            if self.AListener:
                self.AListener.update(Bmsg)

            if VERBOSE > 2:
                if self.AListener:
                    print('B msg %s  upA: p0 %.3f H %.4f Bheard(%s) Bsaid(%s)'
                          % (Bmsg,
                             self.AListener.pr_cat(0),
                             self.AListener.entropy(),
                             ut.prlints(self.AListener.heard_by_learner),
                             ut.prlints(self.AListener.heard_from_learner)))
                else:
                    print('B msg %s  upA: p0 %.3f H %.4f'
                          % (Bmsg,
                             self.ALearner.pr_cat(0),
                             self.ALearner.entropy()))

            if self.AListener:
                Amsg = self.AListener.production(self.s.message_length,
                                                 self.s.accommodation,
                                             ownlang=self.ALearner)
                self.AListener.update_heard_by_learner(Amsg)
            else:
                Amsg = self.ALearner.production(self.s.message_length)
            self.BLearner.update(Amsg)

            if VERBOSE > 2:
                print('A msg %s  upB: H %.4f (%s)' % (
                    Amsg, self.BLearner.entropy(), self.BLearner))

            self.update_Hs(date + dateshift)
            if ps:
                Bps.append(self.BLearner.pr_cat(1))
        if VERBOSE > 0:
            print('FINL A: H %.4f ALearner H %.4f Bheard(%s) Bsaid(%s) B: H %.4f'
                  % (self.AListener.entropy(),
                     self.ALearner.entropy(),
                     ut.prlints(self.AListener.heard_by_learner),
                     ut.prlints(self.AListener.heard_from_learner),
                     self.BLearner.entropy()))

        if ps:
            return (BH, Bps, AH)


def run_iterated_chat(settings, iterations=5, loops=5):
    """Run multiple chats with the same NNS (BLearner)."""

    NNS_Hs = []
    Listener_Hs = []  # speaker model entropy
    for l in range(loops):
        if VERBOSE > 0: print("### Iterative Loop #", l)
        listeners = []  # Concatenation of lots of different NS listeners
        # Chat 1
        c1 = Chat(settings)
        c1.chat()
        NNS = c1.BLearner
        listeners.extend(c1.AListener.H_trajectory)
        # print("NNS",  NNS.H_trajectory)

        for chatiter in range(1, iterations):
            if VERBOSE > 0: print("** Chat #", chatiter)

            cc = Chat(settings, BLearner=NNS)
            cc.chat(dateshift=chatiter * settings.dialogue_length)
            listeners.extend(cc.AListener.H_trajectory)
            # print("NNS", NNS.H_trajectory[-1])

        if VERBOSE > 0:
            print("Final NNS", NNS.H_trajectory)
        NNS_Hs.append(NNS.H_trajectory)
        Listener_Hs.append(listeners)

    with open(settings.outputstr + '.iterchats.NNS_H.csv', 'w') as outputf:
    # one row per NNS (wide form data)
        writer = csv.writer(outputf)
        for row in NNS_Hs:
            writer.writerow([y for x,y in row])  # no dates

    with open(settings.outputstr + '.iterchats.Listener_H.csv', 'w') as outputf:
    # one row per NNS (wide form data)
        writer = csv.writer(outputf)
        for row in Listener_Hs:
            writer.writerow([y for x,y in row])  # no dates


def run_chat(settings):
    """Run a single chat, print the NNS entropy history"""
    c = Chat(settings)
    c.chat()
    print(c.BLearner.H_trajectory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Chat between Bayesian Learners.")
    parser.add_argument('--outputstr', '-o', type=str, default='out.chat')
    parser.add_argument('--language', '-l', type=float, nargs='+')
    parser.add_argument('--alpha_prior', '-a', type=float, default=0.01)
    parser.add_argument('--no_gamma_prior', '-G', type=bool, default=False,
                        help="""Don't use gamma prior: speaker-model datasets
                        have fixed size=D""")
    parser.add_argument('--dialogue_length', '-d', type=int, default=5)
    parser.add_argument('--message_length', '-M', type=int, default=1)
    parser.add_argument('--speaker_hyps', '-S', type=int, default=10,
                        help="""number of components in speaker mod
                        (Listener class in code, confusingly)""")
    parser.add_argument('--prior_data', '-D', type=int, default=2)
    parser.add_argument('--accommodation', '-A', type=float, default=0.0)


    args = parser.parse_args()
    assert args.language is not None  # need a language

    args_dict = vars(args)  # keep args safe
    print(args_dict)

    # too annoying to change in argparse
    if ASSYM_PRIOR:
        args_dict['alpha_prior'] = ASSYM_PRIOR
    print("Language entropy", entropy(args.language, base=2))
    print("Language entropy", entropy(args_dict['alpha_prior'], base=2))

    # XXX testing.
    set_seed()

    settings = ChatSettings(**(args_dict))
    run_iterated_chat(settings, iterations=20, loops=20)
