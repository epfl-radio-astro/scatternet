import numpy as np
import matplotlib.pyplot as plt
import json
with open('results2.json', 'r') as f:
  results = json.load(f)


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
                'cnn':{'linestyle':'-','marker':'*', 'mfc':'magenta',   'c':'magenta'},
                'wavesvm':       {'linestyle':'-', 'marker':'v', 'mfc':'mediumblue',   'c':'mediumblue'},
                'scattersvm':    {'linestyle':'--','marker':'^', 'mfc':'dodgerblue',   'c':'dodgerblue'},
                'redscattersvm': {'linestyle':':', 'marker':'s', 'mfc':'darkturquoise','c':'darkturquoise'},
                'wavenet':       {'linestyle':'-', 'marker':'v', 'mfc':'darkorange',   'c':'darkorange'},
                'scatternet':    {'linestyle':'--','marker':'^', 'mfc':'orange',   'c':'orange'},
                'redscatternet': {'linestyle':':', 'marker':'s', 'mfc':'gold',   'c':'gold'},
}

labels = { 'svm': 'SVM',
            'cnn': 'CNN',
            'wavesvm': 'Wavelets+SVM',
            'scattersvm': 'Scattering2D+SVM',
            'redscattersvm': 'Red. Scattering2D+SVM',
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

    axs[1].errorbar(data[scenario]['x'],data[scenario]['f1']['val'],yerr=data[scenario]['f1']['err'], label = labels[scenario], **f)
    axs[1].set_ylabel( 'F1 score')
    axs[1].set_xlabel( 'Training data size')
plt.legend(ncol =2, bbox_to_anchor=(0.5, 1.5))
plt.show()