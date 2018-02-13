import ccalnoir as ccal
import pandas as pd
import numpy as np
from ccalnoir import differential_gene_expression
from ccalnoir import compute_information_coefficient
import pickle

TOP = 10

RUN = True

# data_file = "test_data/BRCA_cp_40_samples.gct"
data_file = "test_data/BRCA_minimal.gct"
# class_file = "test_data/BRCA_cp_40_samples.cls"
class_file = "test_data/BRCA_minimal.cls"

# data_df = pd.read_table("test_data/BRCA_cp_40_samples.gct", header=2, index_col=0)
# data_df.drop('Description', axis=1, inplace=True)
# temp = open("test_data/BRCA_cp_40_samples.cls")
# temp.readline()
# temp.readline()
# classes = [int(i) for i in temp.readline().strip('\n').split(' ')]
# classes = pd.Series(classes, index=data_df.columns)

if RUN:
    scores = differential_gene_expression(phenotype_file=class_file,
                                          gene_expression=data_file,
                                          output_filename='DE_test',
                                          title='Differential Expression Test',
                                          ranking_method=ccal.custom_pearson_corr,
                                          number_of_permutations=10)

    # pickle.dump(scores, open('match_results.p', 'wb'))
else:
    # scores = pickle.load(open('match_results.p', 'rb'))
    pass

# print(scores.iloc[np.r_[0:TOP, -TOP:0], :])

# scores['abs_score'] = abs(scores['Score'])
# scores['Feature'] = scores.index
# scores.sort_values('abs_score', ascending=False, inplace=True)
# scores.reset_index(inplace=True)
# scores['Rank'] = scores.index + 1
#
# print(scores.iloc[0:2*TOP, :])
