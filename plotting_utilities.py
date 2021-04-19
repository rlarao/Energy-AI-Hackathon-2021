import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as plt_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field(field, name):
    x = np.arange(0, 200, 1)*50
    y = np.arange(0, 200, 1)*50
    X, Y = np.meshgrid(x,y)
    fig, ax = plt.subplots()

    x_fault = [i for i in range(1750,10000,10)]
    y_fault = [11750-i for i in x_fault]

    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.15)
    z_min, z_max = np.min(field) *0.95, np.max(field) * 1.05

    im = ax.pcolormesh(X,Y, field, shading='auto', cmap='turbo', vmin=z_min, vmax=z_max)
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(name)
    #df_prod_hist.plot.scatter(x='X', y='Y', ax=ax, marker='o', c='tab:gray')
    #ax.scatter(df_pre_loc["X"], df_pre_loc["Y"], c="tab:red")
    #scatter = sns.scatterplot(data=data, x='X', y='Y', hue='ai', ax=ax, cmap='inferno', vmin=7e6, vmax=7.7e6 )
    #ax.scatter(data.X, data.Y, c=data[name], cmap='turbo', vmin=z_min, vmax=z_max)
    ax.plot(x_fault, y_fault, "tab:gray", ls = '-', lw=1)
    ax.set_xlim((0, 10000)); ax.set_ylim((0, 10000))
    ax.invert_yaxis()


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='depth')
    cmap_facies = plt_colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.depth.min(); zbot=logs.depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Face_Num'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 12))
    ax[0].plot(logs["por"], logs.depth, '-g')
    ax[1].plot(logs["perm"], logs.depth, '-')
    ax[2].plot(logs["rho"], logs.depth, '-', color='0.5')
    #ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    #ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[3].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=5)
    
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((30*' ').join([' Sand ', 'Sandy-Shale', 'Shaley-Sand', 
                                'Shale','nan']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("por")
    ax[0].set_xlim(logs["por"].min(),logs["por"].max())
    ax[1].set_xlabel("perm")
    ax[1].set_xlim(logs["perm"].min(),logs["perm"].max())
    ax[2].set_xlabel("rho")
    ax[2].set_xlim(logs["rho"].min(),logs["rho"].max())
    #ax[3].set_xlabel("PHIND")
    #ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    #ax[4].set_xlabel("PE")
    #ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[3].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    #ax[4].set_yticklabels([]); #ax[5].set_yticklabels([])
    #ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['ID'], fontsize=14,y=0.94)
    return f,ax

def compare_over_wells(logs, facies_colors,features):
    if not(features in logs[0].columns):
        print("features should be a string matching the column's name")
    #make sure logs are sorted by depth
    logs = [log.sort_values(by='depth') for log in logs]
    cmap_facies = plt_colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop = np.min([log.depth.min() for log in logs])
    zbot = np.max([log.depth.max() for log in logs])
    #ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs[0]['Face_Num'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=len(logs)*2+1, figsize=(8, 12))
    color = ['-g','-','-r']
    im = [1] * (len(logs)*2+1)
    cax = [1] * (len(logs)*2+1)
    cbar = [1] * (len(logs)*2+1)
    divider = [1] * (len(logs)*2+1)
    counter = 0 # loops from 0 to len(logs)*2 
    for i in range(len(logs)): # for each log
        cluster=np.repeat(np.expand_dims(logs[i]['Face_Num'].values,1), 100, 1)
        
        # Plot Feature
        ax[counter].plot(logs[i][features],logs[i].depth,color[i])
        
        ax[counter].set_xlabel(features)
        ax[counter].set_xlim(logs[i][features].min(),logs[i][features].max())
        ax[counter].set_yticklabels([])
        
        # Create color bar for facies
        
        im[counter]=ax[counter+1].imshow(cluster, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin=1,vmax=5)
        divider[counter] = make_axes_locatable(ax[len(logs)*2])
        cax[counter] = divider[counter].append_axes("right", size="20%", pad=0.05)
        cbar[counter]=plt.colorbar(im[counter], cax=cax[counter])
        cbar[counter].set_label((30*' ').join([' Sand ', 'Sandy-Shale', 'Shaley-Sand', 
                                    'Shale','nan']))
        cbar[counter].set_ticks(range(0,1)); 
            
        ax[counter+1].set_xlabel('Facies')
        counter+=2 
        if counter >= len(logs)*2:
            break
    for i in range(len(ax)-1):
        if i%2 == 0:
            ax[i].set_ylim(ztop,zbot)
            ax[i].invert_yaxis()
            ax[i].grid()
            ax[i].locator_params(axis='x', nbins=3)
    f.suptitle('Well: %s - %s'%(logs[0].iloc[0]['ID'],logs[-1].iloc[0]['ID']), fontsize=14,y=0.94)
    return f,ax