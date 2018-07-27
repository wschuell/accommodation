
Code associated with:
Stella Frank and Kenny Smith, 2018.
A model of linguistic accommodation leading to language simplification.
Proceedings of CogSci.


-------------------------
```
usage: ft_multinomial.py [-h] [--outputstr OUTPUTSTR]  
                         [--language LANGUAGE [LANGUAGE ...]]  
                         [--alpha_prior ALPHA_PRIOR]  
                         [--no_gamma_prior NO_GAMMA_PRIOR]  
                         [--dialogue_length DIALOGUE_LENGTH]  
                         [--message_length MESSAGE_LENGTH]  
                         [--speaker_hyps SPEAKER_HYPS]  
                         [--prior_data PRIOR_DATA]  
                         [--accommodation ACCOMMODATION]  

Chat between Bayesian Learners.

optional arguments:
  -h, --help            show this help message and exit  
  --outputstr OUTPUTSTR, -o OUTPUTSTR
  --language LANGUAGE [LANGUAGE ...], -l LANGUAGE [LANGUAGE ...]  
  --alpha_prior ALPHA_PRIOR, -a ALPHA_PRIOR  
  --no_gamma_prior NO_GAMMA_PRIOR, -G NO_GAMMA_PRIOR
                        Don't use gamma prior: speaker-model datasets have
                        fixed size=D
  --dialogue_length DIALOGUE_LENGTH, -d DIALOGUE_LENGTH
  --message_length MESSAGE_LENGTH, -M MESSAGE_LENGTH
  --speaker_hyps SPEAKER_HYPS, -S SPEAKER_HYPS
                        number of components in speaker mod (Listener class in
                        code, confusingly)
  --prior_data PRIOR_DATA, -D PRIOR_DATA
  --accommodation ACCOMMODATION, -A ACCOMMODATION
```