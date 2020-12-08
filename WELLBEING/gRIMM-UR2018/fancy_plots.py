import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fancy_barh(data,formater = lambda x:"{:2.1f}".format(100*x)):


    pos = range(len(data.index))    
    
    #ensures Series are seen as dataFrames
    if  len(data.shape)==1:
        data = pd.DataFrame(data)
        
    nb_cols =  data.shape[1]
    
    has_legend  = nb_cols>1
    
    #orders datafram acording to first columan
    data=data.sort_index(by=data.columns[0])


    #bars
    for col in data.columns:
        plt.barh(pos,data[col],align="center",color="#a1d99b",edgecolor="#31a354",alpha=1/nb_cols)

    #frame
    plt.ylim(ymin=-0.5,ymax = max(pos)+4 if has_legend else  max(pos)+1); #room for legend
    
    if data.min().min() >=0:
        plt.xlim(xmin=0)
    
    ax=plt.gca()
    #remove spines
    ax.spines['bottom'].set_color("none")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')

    #removes xticks
    for tic in ax.yaxis.get_major_ticks()+ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        
    #Yticks
    plt.yticks(pos,data.index);    
        
    #removes xlables
    plt.setp(ax.get_xticklabels(), visible=False); 

    
    #anotations
    is_even   = nb_cols>1
    for col in data.columns:
        
        #All labels
        for i in pos:
            x=data.ix[i,col]
            if x>1/100 or col==data.columns[-1]: #marks big numbers and the last one
                ax.annotate(formater(x),  xy=(x,i),xycoords='data',ha="right" if is_even else "left"
                            ,va="center", size=12,  xytext=(-5 if is_even else 5, -1), textcoords='offset points')
        
        if nb_cols>1 :
            #the "legend"
            ax.annotate(col,  xy=(x,i),xycoords='data',va="center",
              xytext=(-50 if is_even else 50, 35), textcoords='offset points', 
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.3" if is_even else "arc3,rad=0.3",
                                )
                )
            
        is_even = not is_even


from fancy_round import fancy_round
            
def autolabel(ax,rects,color, sigdigits,  **kwargs):
    """attach labels to an existing horizontal bar plot. Passes kwargs to the text (font, color, etc)"""
    
    
    for rect in rects:
        
        #parameters of the rectangle
        h = rect.get_height()
        x = rect.get_x()
        y = rect.get_y()
        w = rect.get_width()
        
        #figures out if it is a negative or positive value
        value = x if x<0 else w

        ####
        # FORMATS LABEL
        
        #truncates the value to sigdigits digits after the coma.
        stri="{v:,f}".format(v=fancy_round(value,sigdigits))
        
        #remove trailing zeros
        if "." in stri:
            while stri.endswith("0"):
                stri=stri[:-1]        
        
        #remove trailing dot
        if stri.endswith("."):
            stri=stri[:-1]        
        
        if stri=="-0":
            stri="0"
        
        #space before or after (pad)
        if value<0:
            stri = stri+' '
        else:
            stri = ' '+stri

        #actual print    
        ax.text(value, y+0.4*h, stri, ha="right" if x<0 else 'left', va='center', color=color , **kwargs)

from matplotlib.ticker import FuncFormatter as funfor

def spine_and_ticks(ax,reverted=False, thousands=False):
    
    if reverted:
    
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color("none")

        #removes ticks 
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        
    else:
        
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color("none")

            #removes ticks 
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        
    if thousands:
        ax.get_xaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
   
def y_thousands_sep(ax=None):
    if ax is None:
        ax=plt.gca()
    ax.get_yaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    
def x_thousands_sep(ax=None):
    if ax is None:
        ax=plt.gca()
        
    ax.get_xaxis().set_major_formatter(funfor(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    
    
def plot_barh(a,color="#9ecae1", xlabel=""):
    #params 
    n=len(a)
    height=.38
    pos=np.arange(n)

    #new fig
    fig, ax = plt.subplots(figsize=(6,n/1.75))
    rects=plt.barh(pos-height/2,(100*a), height=height, color=color, clip_on=False);

    #Department labels
    ax.set_yticks(pos)
    ax.set_yticklabels(a.index)

    #X axis
    ax.xaxis.set_ticklabels([])
#     ax.set_title(xlabel);

    #X labels
    autolabel(ax,rects,"gray",2)
#     ax.set_xlabel("Average error (%)")

    #no spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color("none")

    #no ticks
    for tic in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    plt.tight_layout()
    return ax    
    

import os    
from subprocess import Popen
    
def savefig(path, **kwargs):
    #Saves in both png and pdf (ignores .png and .pdf in path). 
    #passes **kwargs to plt.savefig
    
    # try to solve these layout issues
    try:
        plt.tight_layout()
    except:
        pass
        
    kwargs.update(dict(bbox_inches="tight"))
    
    #ignores png and pdf in file name
    path = path.replace(".png","")
    path = path.replace(".pdf","")

    #saves
    plt.savefig(path+".png", **kwargs)
    # plt.savefig(path+".pdf", **kwargs)
    
    folder = os.path.dirname(path)
    filename  = os.path.basename(path)
    
    # print("path", path, "folder",folder,"filename", filename)
    # aaaaa
    Popen(["convert", filename+".png", "-trim", filename+".png" ], cwd=folder, shell=True )
    # Popen("pdfcrop {fn}.pdf {fn}.pdf".format(fn=filename), cwd=folder )
    
    
    
