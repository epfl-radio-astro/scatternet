import numpy as np
import matplotlib.pyplot as plt
import json, sys



#infile = '/Users/etolley/Desktop/results_minst.json'
#infile = '/Users/etolley/Desktop/results/results_minst/'
#infile = '/Users/etolley/Desktop/results/results_mirabestbinary/'
#infile = '/Users/etolley/Desktop/results/results_aug_mirabest/'
infile = 'results/results_galaxy/'
results = {}

if infile[-5:] == '.json':
    with open('/Users/etolley/Desktop/results_minst.json', 'r') as f:
      results = json.load(f)

else:
    import glob
    flist = glob.glob(infile + '*.json')
    ntrain_samples = [int(fname.split('_')[-1][:-5]) for fname in flist]
    files = zip(ntrain_samples,flist)
    for ntrain, fname in sorted(zip(ntrain_samples,flist), key = lambda i:i[0]):
        with open(fname, 'r') as f:
            filedata = json.load(f)
            for scenario in filedata:
                try:
                    results[scenario][ntrain] = filedata[scenario]
                except:
                    results[scenario] = {}
                    results[scenario][ntrain] = filedata[scenario]
x = []
data = {}

for scenario in results:
    data[scenario] = {'acc':{'val':[], 'err':[]},'f1':{'val':[], 'err':[]}, 'x':[]}
    for ntrain in results[scenario]:
        acc = [results[scenario][ntrain][i]['acc'] for i in results[scenario][ntrain]]
        f1 = [results[scenario][ntrain][i]['f1'] for i in results[scenario][ntrain]]

        print(scenario, ntrain, acc)

        data[scenario]['x'].append(ntrain)
        data[scenario]['acc']['val'].append(np.mean(acc))
        data[scenario]['acc']['err'].append(np.std(acc))
        data[scenario]['f1']['val'].append(np.mean(f1))
        data[scenario]['f1']['err'].append(np.std(f1))


plot_format = { 'svm':{'linestyle':'-','marker':'o', 'mfc':'black', 'c':'black'},
                'cnn':{'linestyle':'--','marker':'*', 'mfc':'magenta',   'c':'magenta'},
                'cnn2':{'linestyle':'-','marker':'x', 'mfc':'violet',   'c':'violet'},
                'wavesvm':       {'linestyle':'-', 'marker':'v', 'mfc':'mediumblue',   'c':'mediumblue'},
                'scattersvm':    {'linestyle':'--','marker':'^', 'mfc':'dodgerblue',   'c':'dodgerblue'},
                'redscattersvm': {'linestyle':':', 'marker':'s', 'mfc':'darkturquoise','c':'darkturquoise'},
                'wavenet':       {'linestyle':'-', 'marker':'v', 'mfc':'darkorange',   'c':'darkorange'},
                'scatternet':    {'linestyle':'--','marker':'^', 'mfc':'orange',   'c':'orange'},
                'redscatternet': {'linestyle':':', 'marker':'s', 'mfc':'gold',   'c':'gold'},
}

labels = { 'svm': 'SVC',
            'cnn': 'CNN1',
            'cnn2': 'CNN2',
            'wavesvm': 'Wavelets+SVC',
            'scattersvm': 'Scattering2D+SVC',
            'redscattersvm': 'Red. Scattering2D+SVC',
            'wavenet':       'Wavelets+NN',
            'scatternet':    'Scattering2D+NN',
            'redscatternet':  'Red. Scattering2D+NN',
    
}

fig, axs = plt.subplots(1,2, figsize=(8, 4))
plt.subplots_adjust(left=0.1, right=0.95, top=0.7, bottom=0.15)
for scenario in results:
    f = {}
    if scenario in plot_format:
        f = plot_format[scenario]
    axs[0].errorbar(data[scenario]['x'],data[scenario]['acc']['val'],yerr=data[scenario]['acc']['err'], **f)
    axs[0].set_ylabel( 'Accuracy')
    axs[0].set_xlabel( 'Training data size')
    axs[0].set_xscale('log')

    axs[1].errorbar(data[scenario]['x'],data[scenario]['f1']['val'],yerr=data[scenario]['f1']['err'], label = labels[scenario], **f)
    axs[1].set_ylabel( 'F1 score')
    axs[1].set_xlabel( 'Training data size')
    axs[1].set_xscale('log')
plt.legend(ncol =2, bbox_to_anchor=(0.5, 1.5))
plt.show()