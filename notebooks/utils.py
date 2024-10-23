'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import os
import pickle
import time
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve, confusion_matrix

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

palette=sns.color_palette(['#023047', '#e85d04', '#0077b6', '#ff8200', '#0096c7', '#ff9c33'])

def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                   outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
                   nominal=False, color='#023047', figsize=(24, 12)):

    # Get num_features and num_rows and iterating over the sublot dimensions.
    num_features = len(features)
    num_rows = num_features // 3 + (num_features % 3 > 0) 
    
    fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

    for i, feature in enumerate(features):
        row = i // 3  
        col = i % 3  

        ax = axes[row, col] if num_rows > 1 else axes[col] 
        
        if barplot:
            if mean:
                data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                data_grouped[mean] = round(data_grouped[mean], 2)
                ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                for index, value in enumerate(data_grouped[mean]):
                    # Adjust the text position based on the width of the bars
                    ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
            else:
                if hue:
                    data_grouped = data.groupby([feature])[[hue]].mean().reset_index().rename(columns={hue: 'pct'})
                    data_grouped['pct'] *= 100
                else:
                    data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                    data_grouped['pct'] = data_grouped['count'] / data_grouped['count'].sum() * 100
    
                ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                
                if pd.api.types.is_numeric_dtype(data_grouped[feature]):
                    ax.invert_yaxis()
                    
                for index, value in enumerate(data_grouped['pct']):
                    # Adjust the text position based on the width of the bars
                    ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
            
            ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
            ax.get_xaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
    
        elif outliers:
            # Plot univariate boxplot.
            sns.boxplot(data=data, x=feature, ax=ax, color=color)
        
        elif boxplot:
            # Plot multivariate boxplot.
            sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)

        else:
            # Plot histplot.
            sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue)

        ax.set_title(feature)  
        ax.set_xlabel('')  
    
    # Remove unused axes.
    if num_features < len(axes.flat):
        for j in range(num_features, len(axes.flat)):
            fig.delaxes(axes.flat[j])

    plt.tight_layout()


def check_outliers(data, features):
    
    outlier_counts = {}
    outlier_indexes = {}
    total_outliers = 0
    
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_indexes[feature] = feature_outliers.index.tolist()
        outlier_count = len(feature_outliers)
        outlier_counts[feature] = outlier_count
        total_outliers += outlier_count
    
    print(f'There are {total_outliers} outliers in the dataset.')
    print()
    print(f'Number (percentage) of outliers per feature: ')
    print()
    for feature, count in outlier_counts.items():
        print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

    return outlier_indexes, outlier_counts, total_outliers


def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    # Stratified KFold
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Dicionários para armazenar as pontuações
    models_val_scores = dict()
    models_train_scores = dict()
    models_f1_scores = dict()
    models_recall_scores = dict()

    for model in models:
        model_instance = models[model]

        # Medir o tempo de treinamento
        start_time = time.time()
        
        # Ajustar o modelo
        model_instance.fit(X_train, y_train)

        end_time = time.time()
        training_time = end_time - start_time

        # Previsões nos dados de treinamento
        y_train_pred = model_instance.predict(X_train.values)
        train_score = roc_auc_score(y_train, y_train_pred)

        # Avaliar o modelo com k-fold cross validation
        val_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='roc_auc', cv=stratified_kfold)
        avg_val_score = val_scores.mean()
        val_score_std = val_scores.std()

        # Calcular F1 Score e Recall
        f1_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='f1', cv=stratified_kfold)
        avg_f1_score = f1_scores.mean()

        recall_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='recall', cv=stratified_kfold)
        avg_recall_score = recall_scores.mean()

        # Armazenar os resultados
        models_val_scores[model] = avg_val_score
        models_train_scores[model] = train_score
        models_f1_scores[model] = avg_f1_score
        models_recall_scores[model] = avg_recall_score

        # Imprimir os resultados
        print(f'{model} results: ')
        print('-'*50)
        print(f'Training score: {train_score}')
        print(f'Average validation score: {avg_val_score}')
        print(f'Standard deviation: {val_score_std}')
        print(f'Average F1 Score: {avg_f1_score}')
        print(f'Average Recall Score: {avg_recall_score}')
        print(f'Training time: {round(training_time, 5)} seconds')
        print()

    # Compilar os resultados em um DataFrame
    train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
    val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
    f1_df = pd.DataFrame(list(models_f1_scores.items()), columns=['model', 'avg_f1_score'])
    recall_df = pd.DataFrame(list(models_recall_scores.items()), columns=['model', 'avg_recall_score'])

    # Merge dos DataFrames
    eval_df = val_df.merge(train_df, on='model').merge(f1_df, on='model').merge(recall_df, on='model')

    # Ordenar pelo melhor ROC-AUC
    eval_df = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
    
    return eval_df


def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):

    # Plot each model and their train and validation (average) scores.
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(eval_df['model']))
    y = np.arange(len(eval_df['train_score']))

    val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color='#023047')
    train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color='#0077b6')

    ax.set_xlabel('Modelo', labelpad=labelpad, fontsize=label_size)
    ax.set_ylabel('ROC-AUC', labelpad=labelpad, fontsize=label_size)
    ax.set_title("Desempenho dos Modelos", fontweight='bold', fontsize=title_size, pad=title_pad)
    ax.set_xticks(x, eval_df['model'], rotation=0, fontsize=10.8)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.tick_params(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    # Add scores on top of each bar
    for bar in val_bars + train_bars:
        height = bar.get_height()
        plt.annotate('{}'.format(round(height, 2)),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom')

    # Define handles and labels for the legend with adjusted sizes
    handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor = 'none'),
            plt.Rectangle((0,0), 0.1, 0.1, fc='#0077b6', edgecolor = 'none')]
    labels = ['Val score', 'Train score']
        
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)


def evaluate_classifier(y_true, y_pred, probas):

    # Print classification report and calculate its metrics to include in the final metrics df.
    print(classification_report(y_true, y_pred))
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    roc_auc = roc_auc_score(y_true, probas)
        
    # Confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Real Values')
    plt.show()
    
    # Plot ROC Curve and ROC-AUC.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color='#023047')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#e85d04')  # Random guessing line.
    ax.set_xlabel('False Positive Rate', fontsize=10.8, labelpad=20, loc='left')
    ax.set_ylabel('True Positive Rate', fontsize=10.8, labelpad=20, loc='top')
    ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_title('Receiver operating characteristic (ROC) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    # PR AUC Curve and score.

    # Calculate model precision-recall curve.
    p, r, _ = precision_recall_curve(y_true, probas)
    pr_auc = auc(r, p)
    
    # Plot the model precision-recall curve.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, p, marker='.', label=f'PR AUC = {pr_auc:.2f}', color='#023047')
    ax.set_xlabel('Recall', fontsize=10.8, labelpad=20, loc='left')
    ax.set_ylabel('Precision', fontsize=10.8, labelpad=20, loc='top')
    ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_title('Precision-recall (PR) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    # Construct a DataFrame with metrics for passed sets.
    model_metrics = pd.DataFrame({
                                'Metric': ['Precision','Recall','F1-Score','ROC-AUC','PR-AUC'],
                                'Value': [precision,recall,f1,roc_auc,pr_auc],
                                })
    
    return model_metrics


def plot_probability_distributions(y_true, probas):

    probas_df = pd.DataFrame({'churn_probability': probas,
                            'churn': y_true})

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=probas_df, x='churn_probability', hue='churn', fill=True, ax=ax, palette=['#023047', '#e85d04'])
    ax.set_title('Distribuição da probabilidade de cancelamento', fontweight='bold', fontsize=12, pad=45, loc='left')
    ax.set_xlabel('Probabilidades previstas', fontsize=10.8, labelpad=20, loc='left')
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                )
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    
    handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#e85d04', edgecolor = 'none'),
            plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor = 'none')]
    labels = ['Cancela', 'Não Cancela']
        
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.14, 1.15), frameon=False, ncol=2, fontsize=10)


def plot_feature_importances(model, data):

    # Get feature importances
    importances = model.feature_importances_
    feature_names = data.columns 


    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 3))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names, color='#023047')
    plt.xticks(rotation=90)
    plt.show()


def save_object(file_path, object):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'wb') as file_object:
        pickle.dump(object, file_object)
