import numpy as np
import matplotlib.pyplot as plt
import json, sys

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

def load_noise_data(infile):
    with open(infile, 'r') as f:
      results = json.load(f)

    data = {}

    for scenario in results:
        data[scenario] = {'acc':{'val':[], 'err':[]},'f1':{'val':[], 'err':[]}, 'x':[]}
        for noise in results[scenario]:
            data[scenario]['x'].append(noise)

            acc =  [results[scenario][noise][i]['acc'] for i in results[scenario][noise]]
            f1  =  [results[scenario][noise][i]['f1']  for i in results[scenario][noise]]

            print(scenario, noise, acc)

            data[scenario]['acc']['val'].append(np.mean(acc))
            data[scenario]['acc']['err'].append(np.std(acc))
            data[scenario]['f1']['val'].append(np.mean(f1))
            data[scenario]['f1']['err'].append(np.std(f1))
    return data

def load_size_data(infile):
    results = {}

    if infile[-5:] == '.json':
        with open(infile, 'r') as f:
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
    return data



def make_noise_plot(infile):

    data = load_noise_data(infile)

    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.7, bottom=0.15)
    for scenario in data:
        f = {}
        if scenario in plot_format:
            f = plot_format[scenario]
        axs[0].errorbar(data[scenario]['x'],data[scenario]['acc']['val'],yerr=data[scenario]['acc']['err'], **f)
        axs[0].set_ylabel( 'Accuracy')
        axs[0].set_xlabel( 'Injected noise')
        #axs[0].set_xscale('log')

        axs[1].errorbar(data[scenario]['x'],data[scenario]['f1']['val'],yerr=data[scenario]['f1']['err'], label = labels[scenario], **f)
        axs[1].set_ylabel( 'F1 score')
        axs[1].set_xlabel( 'Injected noise')
        #axs[1].set_xscale('log')
    plt.legend(ncol =2, bbox_to_anchor=(0.5, 1.5))
    plt.show()

def make_noise_plots(infile1, infile2, infile3):

    data1 = load_noise_data(infile1)
    data2 = load_noise_data(infile2)
    data3 = load_noise_data(infile3)

    fig, axs = plt.subplots(1,3, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.7, bottom=0.15, wspace=0)
    for scenario in data1:
        f = {}
        if scenario in plot_format:
            f = plot_format[scenario]
        axs[0].errorbar(data1[scenario]['x'],data1[scenario]['acc']['val'],yerr=data1[scenario]['acc']['err'], **f)
        axs[1].errorbar(data2[scenario]['x'],data2[scenario]['f1']['val'],yerr=data2[scenario]['f1']['err'], label = labels[scenario], **f)
        axs[2].errorbar(data3[scenario]['x'],data3[scenario]['acc']['val'],yerr=data3[scenario]['acc']['err'], **f)

        
        #axs[1].set_xscale('log')
    axs[1].legend(ncol =2, bbox_to_anchor=(0.5, 1.5))

    for i in range(3):
        axs[i].set_ylim(0.0,1.0)
        axs[i].grid(axis = 'y')
        axs[i].set_xlabel( 'Injected noise')

    axs[0].set_ylabel( 'Accuracy')
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])


    
    plt.show()



def make_train_size_plot(infile):

    data = load_size_data(infile)

    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.7, bottom=0.15)
    for scenario in data:
        f = {}
        if scenario in plot_format:
            f = plot_format[scenario]
        axs[0].errorbar(data[scenario]['x'],data[scenario]['acc']['val'],yerr=data[scenario]['acc']['err'], **f)
        axs[0].set_ylabel( 'Accuracy')
        axs[0].set_xlabel( 'Training data size')
        axs[0].set_xscale('log')
        axs[0].grid(axis = 'y')

        axs[1].errorbar(data[scenario]['x'],data[scenario]['f1']['val'],yerr=data[scenario]['f1']['err'], label = labels[scenario], **f)
        axs[1].set_ylabel( 'F1 score')
        axs[1].set_xlabel( 'Training data size')
        axs[1].set_xscale('log')
        axs[1].grid(axis = 'y')
    plt.legend(ncol =2, bbox_to_anchor=(0.5, 1.5))
    plt.show()

def compare_augmentation(infile1, infile2):
    data1 = load_size_data(infile1)
    data2 = load_size_data(infile2)

    print(data1.keys())

    fig, axs = plt.subplots(2,1, figsize=(8, 5),gridspec_kw={'height_ratios':[3,1]})
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,hspace=0)

    scenarios = ['cnn', 'cnn2', 'svm', 'scattersvm','scatternet']
    for scenario in scenarios:
        f = plot_format[scenario]
        f['linestyle'] = '-'
        y1 = np.array(data1[scenario]['acc']['val'])
        y2 = np.array(data2[scenario]['acc']['val'])
        e1 = np.array(data1[scenario]['acc']['err'])
        e2 = np.array(data2[scenario]['acc']['err'])
        axs[1].errorbar(data1[scenario]['x'],y1/y2,yerr=( (y1/y2)**2*( (e1/y1)**2 + (e2/y2)**2 ))**0.5, label = labels[scenario],**f)
        axs[1].set_ylabel( 'Augmented/Original')
        axs[0].errorbar(data1[scenario]['x'],y1,yerr=e1, **f)
        f['linestyle'] = ':'
        axs[0].errorbar(data2[scenario]['x'],y2,yerr=e2, **f)
    axs[0].set_ylabel( 'Accuracy')
    axs[1].set_xlabel( 'Training data size')
    axs[0].set_ylim(0.1,1.0)
    #axs[1].set_ylim(0.9,1.31)
    axs[1].plot([],[], c='gray', ls ='-', label="Augmented Data")
    axs[1].plot([],[], c='gray', ls =':', label="Original Data")
    axs[1].grid(axis = 'y')
    axs[0].grid(axis = 'y')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    
    #plt.legend(ncol =2,bbox_to_anchor=(0.6, 2.0))
    plt.legend(ncol =2,bbox_to_anchor=(0.55, 4.0))
    plt.show()


def compare_datasets(infile1, infile2):
    data1 = load_size_data(infile1)
    data2 = load_size_data(infile2)

    print(data1.keys())

    fig, axs = plt.subplots(1,2, figsize=(12, 5))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,wspace=0)

    #scenarios = ['svm', 'scattersvm','redscattersvm','wavesvm']
    scenarios = ['cnn', 'cnn2','scatternet','redscatternet','wavenet']
    #scenarios += ['scatternet','redscatternet','wavenet']
    for scenario in scenarios:
        f = plot_format[scenario]
        #f['linestyle'] = '-'
        y1 = np.array(data1[scenario]['acc']['val'])
        y2 = np.array(data2[scenario]['acc']['val'])
        e1 = np.array(data1[scenario]['acc']['err'])
        e2 = np.array(data2[scenario]['acc']['err'])

        axs[0].errorbar(data1[scenario]['x'],y1,yerr=e1, **f)
        axs[1].errorbar(data2[scenario]['x'],y2,yerr=e2, **f, label = labels[scenario])
    axs[1].legend(ncol =2)
    axs[0].set_ylabel( 'Accuracy')
    axs[0].set_xlabel( 'Training data size')
    axs[1].set_xlabel( 'Training data size')
    axs[0].set_ylim(0.4,1.0)
    axs[1].set_ylim(0.4,1.0)
    axs[1].set_yticklabels([])

    #axs[1].set_ylim(0.9,1.31)
    #axs[1].plot([],[], c='gray', ls ='-', label="Augmented Data")
    #axs[1].plot([],[], c='gray', ls =':', label="Original Data")
    axs[1].grid(axis = 'y')
    axs[0].grid(axis = 'y')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    
    #plt.legend(ncol =2,bbox_to_anchor=(0.6, 2.0))
    plt.savefig("summary_nn.pdf")
    plt.show()

#make_noise_plot("/Users/etolley//Desktop/results_noise/results_minst/results_minst.json")
#make_noise_plot("/Users/etolley/Desktop/results_aug_mirabestbinary.json")
#make_noise_plot("./results_noise/results_aug_galaxy/results_aug_galaxy.json")
make_noise_plots("/Users/etolley/Desktop/results_aug_mirabestbinary.json",
                 "./results_noise/results_aug_galaxy/results_aug_galaxy.json",
                 "/Users/etolley//Desktop/results_noise/results_minst/results_minst.json")
#make_train_size_plot("results/results_aug_galaxy/")
#make_train_size_plot("/Users/etolley/Desktop/results/results_minst/")
#compare_augmentation("results/results_aug_galaxy/","results/results_galaxy/")
#compare_augmentation("/Users/etolley/Desktop/results/results_aug_mirabestbinary/","/Users/etolley/Desktop/results/results_mirabestbinary/")
#compare_datasets("/Users/etolley/Desktop/results/results_aug_mirabestbinary/","results/results_aug_galaxy/")