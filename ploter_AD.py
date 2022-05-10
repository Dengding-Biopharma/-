from PC1plot import pc1
from PCAplot import pca
from PLSDA import plsda
from boxplot import boxplot
from heatmap import heatmap
from randomforestImportance import randomforestimportance
from stackedHistgramForTOP20 import stackedHistgramTop20
from stackedHistgramForlowest20Pvalue import stackedHistogram

file_name = 'files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx'

plsda(file_name)
pca(file_name)
boxplot(file_name)
pc1(file_name)
randomforestimportance(file_name)
stackedHistogram(file_name,20)
stackedHistgramTop20(file_name)





heatmap(file_name)

