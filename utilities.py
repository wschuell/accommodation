#!/usr/bin/python3


def str_listfs(floats):
    """pretty-print a list of floats"""
    return ", ".join(['%.3f' % f for f in floats])


def listfs1(floats):
    """pretty-print a list of floats"""
    return '[' + ', '.join(['%.1f' % f for f in floats]) + ']'


def listfs2(floats):
    """pretty-print a list of floats"""
    return '[' + ', '.join(['%.2f' % f for f in floats]) + ']'


def str_listints(floats):
    """pretty-print a list of ints"""
    return ", ".join(['%d' % f for f in ints])


def prlints(lints):
    """pretty-print a list of ints"""
    return ", ".join(['%d' % i for i in lints])
