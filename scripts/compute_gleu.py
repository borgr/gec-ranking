#!/usr/bin/env python

# Courtney Napoles
# <courtneyn@jhu.edu>
# 21 June 2015
# ##
# compute_gleu
#
# This script calls gleu.py to calculate the GLEU score of a sentence, as
# described in our ACL 2015 paper, Ground Truth for Grammatical Error
# Correction Metrics by Courtney Napoles, Keisuke Sakaguchi, Matt Post,
# and Joel Tetreault.
#
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# This script was adapted from compute-bleu by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>
#
# Example:
# python gec_ranking/scripts/compute_gleu.py -r data/references/NUCLEB -s data/paragraphs/conll.tok.orig -o data/paragraphs/JMGR -d

import argparse
import sys
import os
from gleu import GLEU
import scipy.stats
import numpy as np
import random
import six

def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
    return ['%f' % mean,
            '%f' % std,
            '(%.3f,%.3f)' % (ci[0], ci[1])]

def gleu_scores(source, references, systems, ngrams_len=4, num_iterations=500, debug=False):
    # if there is only one reference, just do one iteration
    if len(references) == 1:
        num_iterations = 1

    gleu_calculator = GLEU(ngrams_len)

    if isinstance(source, six.string_types):
        gleu_calculator.load_sources(source)
    else:
        gleu_calculator.set_sources(source)

    if isinstance(references[0], six.string_types):
        gleu_calculator.load_references(references)
    else:
        gleu_calculator.set_references(references)

    total = []
    per_sentence = []
    for hpath in systems:
        if isinstance(hpath, six.string_types):
            with open(hpath) as instream:
                hyp = [line.split() for line in instream]
            if not debug:
                print(os.path.basename(hpath),)
        else:
            instream = hpath
            hyp = [line.split() for line in instream]

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, len(references) - 1)
                            for i in range(len(hyp))])

        if debug:
            print()
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [[0 for i in range(2 * ngrams_len + 2)]
                      for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(len(references))]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in gleu_calculator.gleu_stats(
                        i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores)
                                 for scores in zip(iter_stats[j], this_stats)]

            per_sentence.append(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True)
                                                for stats in stats_by_ref]))
            if debug:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(len(references)):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(
                            i, r_ind=r)]

                print(i,)
                print(' '.join(per_sentence[-1]))
        total.append(get_gleu_stats([gleu_calculator.gleu(stats)
                                     for stats in iter_stats]))
        if debug:
            print('\n==== Overall score =====')
            print('Mean Stdev 95%CI GLEU')
            print(' '.join(total[-1]))
        else:
            print("total", total[-1][0])
    return total, per_sentence


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference",
                        help="Target language reference sentences. Multiple "
                        "files for multiple references.",
                        nargs="*",
                        dest="reference",
                        required=True)
    parser.add_argument("-s", "--source",
                        help="Source language source sentences",
                        dest="source",
                        required=True)
    parser.add_argument("-o", "--hypothesis",
                        help="Target language hypothesis sentences to evaluate "
                        "(can be more than one file--the GLEU score of each "
                        "file will be output separately). Use '-o -' to read "
                        "hypotheses from stdin.",
                        nargs="*",
                        dest="hypothesis",
                        required=True)
    parser.add_argument("-n",
                        help="Maximum order of ngrams",
                        type=int,
                        default=4)
    parser.add_argument("-d", "--debug",
                        help="Debug; print sentence-level scores",
                        default=False,
                        action="store_true")
    parser.add_argument('--iter',
                        type=int,
                        default=500,
                        help='the number of iterations to run')

    args = parser.parse_args()
    gleu_scores(args.source, args.reference, args.hypothesis, args.n, num_iterations=args.iter, debug=args.debug)