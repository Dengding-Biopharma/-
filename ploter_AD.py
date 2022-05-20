from PC1plot import pc1
from PCAplot import pca
from PLSDA import plsda
from boxplot import boxplot
from heatmap import heatmap
from randomforestImportance import randomforestimportance
from stackedHistgramForTOP20 import stackedHistgramTop20
from stackedHistgramForlowest20Pvalue import stackedHistogram

mode = 'BOTH'
if mode == 'BOTH':
    file_name = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
elif mode == 'POS':
    file_name = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
elif mode == 'NEG':
    file_name = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'

plsda(file_name,mode)
pca(file_name,mode)
boxplot(file_name,mode)
pc1(file_name,mode)
randomforestimportance(file_name,mode)
stackedHistogram(file_name,k=20,mode=mode)
stackedHistgramTop20(file_name,mode)
heatmap(file_name,mode)

