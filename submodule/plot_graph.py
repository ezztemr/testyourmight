import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_graph(data):
    cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
    con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
    
    # --------------------------------------------- fig1 -----------------------------------------------------------
    fig1 = plt.figure(figsize=(18, 15))
    text_labels = ['Sex', 'Exng', 'Caa', 'Cp', 'Fbs', 'Restecg', 'Slp', 'Thall']
    background_color = "#ffe6e6"
    color_palette = ["#800000", "#8000ff", "#6aac90", "#5833ff", "#da8829"]
    fig1.patch.set_facecolor(background_color)
    
    ax0 = plt.subplot(3, 3, 1)
    ax0.set_facecolor(background_color)
    ax0.text(0.5, 0.5, 'Count plot for various\n categorical features\n_________________',
             horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', 
             fontfamily='serif', color="#000000")
    ax0.axis('off')
    
    for i, variable in enumerate(cat_cols):
        label = text_labels[i]
        ax = plt.subplot(3, 3, i + 2)
        ax.set_facecolor(background_color)
        ax.text(0.5, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight='bold', fontfamily='serif', color="#000000",
                horizontalalignment='center')
        ax.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        sns.countplot(data=data, x=variable, palette=color_palette, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # --------------------------------------------- fig2 -----------------------------------------------------------
    fig2 = plt.figure(figsize=(18, 16))
    gs2 = fig2.add_gridspec(2, 3)
    fig2.patch.set_facecolor(background_color)
    ax1 = fig2.add_subplot(gs2[0, 0])
    ax1.set_facecolor(background_color)
    ax1.text(0.5, 0.5, 'Boxen plot for various\n continuous features\n_________________',
             horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', 
             fontfamily='serif', color="#000000")
    ax1.axis('off')

    text_labels2 = ['Age', 'Trtbps', 'Chol', 'Thalachh', 'Oldpeak']
    for i, variable in enumerate(con_cols):
        label = text_labels2[i]
        row, col = divmod(i + 1, 3)
        ax = fig2.add_subplot(gs2[row, col])
        ax.set_facecolor(background_color)
        ax.text(0.5, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight='bold', fontfamily='serif', color="#000000",
                horizontalalignment='center')
        ax.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        sns.boxenplot(ax=ax, y=data[variable], palette=[color_palette[i]], width=0.6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # --------------------------------------------- fig3 -----------------------------------------------------------
    fig3 = plt.figure(figsize=(18, 7))
    gs3 = fig3.add_gridspec(1, 2)
    fig3.patch.set_facecolor(background_color)
    ax0 = fig3.add_subplot(gs3[0, 0])
    ax0.set_facecolor(background_color)
    ax0.text(0.5, 0.5, "Count of the target\n___________",
             horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', 
             fontfamily='serif', color="#000000")
    ax0.axis('off')

    ax1 = fig3.add_subplot(gs3[0, 1])
    ax1.set_facecolor(background_color)
    ax1.text(0.35, 177, "Output", fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
    ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.countplot(ax=ax1, data=data, x='output', palette=color_palette)
    ax1.set_xticklabels(["Low chances of attack(0)", "High chances of attack(1)"])
    for spine in ["top", "left", "bottom", "right"]:
        ax1.spines[spine].set_visible(False)

    # --------------------------------------------- fig4 -----------------------------------------------------------
    fig4 = plt.figure(figsize=(10, 10))
    gs4 = fig4.add_gridspec(1, 1)
    ax0 = fig4.add_subplot(gs4[0, 0])
    data_corr = data[con_cols].corr()
    mask = np.triu(np.ones_like(data_corr), k=1)
    ax0.text(1.5, -0.1, "Correlation Matrix", fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
    sns.heatmap(data_corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', ax=ax0)

    # --------------------------------------------- fig5 -----------------------------------------------------------
    fig5 = plt.figure(figsize=(12, 12))
    corr_mat = data.corr().stack().reset_index(name="correlation")
    g = sns.relplot(data=corr_mat, x="level_0", y="level_1", hue="correlation", size="correlation",
                    palette="coolwarm", hue_norm=(-1, 1), edgecolor=".7", height=10, sizes=(50, 250), 
                    size_norm=(-.2, .8))
    g.set(xlabel="features on X", ylabel="features on Y", aspect="equal")
    g.fig.suptitle('Scatterplot Heatmap', fontsize=22, fontweight='bold', fontfamily='serif', color="#000000")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    fig5 = g.fig

    # --------------------------------------------- fig6 -----------------------------------------------------------
    fig6 = plt.figure(figsize=(18, 18))
    gs6 = fig6.add_gridspec(5, 2)
    fig6.patch.set_facecolor(background_color)
    text_labels3 = [
        'Distribution of age\naccording to\n target variable\n___________',
        'Distribution of trtbps\naccording to\n target variable\n___________',
        'Distribution of chol\naccording to\n target variable\n___________',
        'Distribution of thalachh\naccording to\n target variable\n___________',
        'Distribution of oldpeak\naccording to\n target variable\n___________'
    ]
    for i, variable in enumerate(con_cols):
        label = text_labels3[i]
        ax_title = fig6.add_subplot(gs6[i, 0])
        ax_title.set_facecolor(background_color)
        ax_title.text(0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', 
                      fontsize=18, fontweight='bold', fontfamily='serif', color='#000000')
        ax_title.axis('off')

        ax_plot = fig6.add_subplot(gs6[i, 1])
        ax_plot.set_facecolor(background_color)
        ax_plot.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        sns.kdeplot(ax=ax_plot, data=data, x=variable, hue="output", fill=True, palette=color_palette, alpha=.5, linewidth=0)
        for spine in ax_plot.spines.values():
            spine.set_visible(False)
        ax_plot.set_xlabel("")
        ax_plot.set_ylabel("")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # --------------------------------------------- fig7 -----------------------------------------------------------
    fig7 = plt.figure(figsize=(18, 20))
    gs7 = fig7.add_gridspec(6, 2)
    fig7.patch.set_facecolor(background_color)
    titles_and_descriptions = [
        ("Chest pain\ndistribution\n__________", "0 - Typical Angina\n1 - Atypical Angina\n2 - Non-anginal Pain\n3 - Asymptomatic"),
        ("Number of\nmajor vessels\n___________", "0 vessels\n1 vessel\n2 vessels\n3 vessels\n4 vessels"),
        ("Heart Attack\naccording to\nsex\n______", "0 - Female\n1 - Male"),
        ("Distribution of thall\naccording to\n target variable\n___________", "Thalium Stress\nTest Result\n0, 1, 2, 3"),
        ("Boxen plot of\nthalachh wrt\noutcome\n_______", "Maximum heart\nrate achieved"),
        ("Strip Plot of\nexng vs age\n______", "Exercise induced\nangina\n0 - No\n1 - Yes")
    ]
    plots_info = [
        ('cp', sns.kdeplot, {'hue': 'output', 'fill': True, 'palette': color_palette, 'alpha': 0.5, 'linewidth': 0}),
        ('caa', sns.kdeplot, {'hue': 'output', 'fill': True, 'palette': color_palette, 'alpha': 0.5, 'linewidth': 0}),
        ('sex', sns.countplot, {'hue': 'output', 'palette': color_palette}),
        ('thall', sns.kdeplot, {'hue': 'output', 'fill': True, 'palette': color_palette, 'alpha': 0.5, 'linewidth': 0}),
        ('output', sns.boxenplot, {'y': 'thalachh', 'palette': color_palette}),  
        ('age', sns.stripplot, {'x': 'exng', 'hue': 'output', 'palette': color_palette, 'jitter': True})  
    ]
    for i, (title_text, description) in enumerate(titles_and_descriptions):
        ax_title = fig7.add_subplot(gs7[i, 0])
        ax_title.set_facecolor(background_color)
        ax_title.text(0.5, 0.5, title_text, horizontalalignment='center', verticalalignment='center', 
                      fontsize=18, fontweight='bold', fontfamily='serif', color='#000000')
        ax_title.text(1, 0.5, description, horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax_title.axis('off')
    
    for i, (variable, plot_func, plot_kwargs) in enumerate(plots_info):
        ax_plot = fig7.add_subplot(gs7[i, 1])
        ax_plot.set_facecolor(background_color)
        ax_plot.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        if plot_func == sns.stripplot:
            plot_func(ax=ax_plot, data=data, y=variable, **plot_kwargs)
            ax_plot.set_xlabel('exng', fontsize=12)
            ax_plot.set_ylabel('age', fontsize=12)
        else:
            plot_func(ax=ax_plot, data=data, x=variable, **plot_kwargs)
            ax_plot.set_xlabel("")
            ax_plot.set_ylabel("")
        for spine in ax_plot.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7