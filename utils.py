
def save_acc_pkl(save_path, training_history):
    import pandas as pd
    eval_df = pd.DataFrame(columns=('task_id', 'majority_acc', 'minority_acc', 'model', 'epoch', 'lr', 'batchsize'))
    for task_id, value in training_history.items():
        line_df = [{'task_id':task_id, \
            'majority_acc': value[0]['majority_eval_acc'], \
            'minority_acc': value[0]['minority_eval_acc'], \
            'model': value[0]['args']['model'], \
            'epoch': value[0]['args']['epochs'], \
            'lr': value[0]['args']['lr'], \
            'batchsize': value[0]['args']['batch_size'] }]
        eval_df = eval_df.append(line_df)
    eval_df.to_pickle(save_path)
    return eval_df

def plot_acc(save_path, df_data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    color_list = ['crimson','coral','burlywood','skyblue', 'royalblue']
    
    plt.figure(figsize=(6,6))
    plt.grid()
    sns.scatterplot(data=df_data, x='majority_acc', y='minority_acc', hue='model',\
                    palette=color_list[:len(df_data['model'].unique())], alpha=0.7) 
    plt.xlabel('Majority Subpopulation Accuracy', fontsize=15)
    plt.ylabel('Minority Subpopulation Accuracy', fontsize=15)
    plt.plot([0,1], [0,1], color='darkgrey', linewidth=1.5, linestyle='--')
    plt.legend(fontsize=12, title_fontsize=15, title='Model')
    plt.savefig(save_path, dpi=500)
    plt.close()