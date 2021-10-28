#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')
    

def get_counts(df, class_col_name):
    counts = {key:0 for key in classes}
    for key in counts.keys():
        counts[key] = df[df[class_col_name] == key].shape[0]
    return counts


def graficoPizzas(data, keys, titulo, nomeArquivoSaida):

    # Aqui criamos area que plotamos o grafico e definimos seu tamanho
    _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
    
    # Criando o grafico e colocando a legenda interna
    wedges, _, autotexts = ax.pie(data, autopct=lambda pct: "{:.1f}%".format(pct),
                                      textprops=dict(color="w"))
    
    # Definindo a caixa de legenda externa, titulo, localizacao e onde vai 'ancorar o box'
    ax.legend(wedges, keys,
              title="Requisito",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Aqui definimos o tamanho do texto de dentro do grafico, e o peso da fonte como bold
    plt.setp(autotexts, size=8, weight="bold")
    
    # Titulo do grafico
    ax.set_title(titulo)
    
    plt.tight_layout(rect=(0, 0, 0, 0))
    plt.savefig("./figures/" + nomeArquivoSaida)
    
    # Mostrando o grafico
    plt.show()
    
    plt.close()
    
    
def graficoBarras(data, keys, indeces, titulo, nomeArquivoSaida):
    
    plt.bar(indeces, data, width=0.45)
    # draw plot for all classes

    plt.title(titulo)
    plt.xlabel('Requisito')
    plt.ylabel('Quantidade')
    plt.xticks(indeces, keys)
    plt.yticks(np.arange(0, 600, 50))
    
    plt.savefig("./figures/" + nomeArquivoSaida)
    
    plt.show()
    plt.close()


if __name__ == '__main__':
    
    if not path.exists('figures/'):
        makedirs('figures/')

    classes = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
    
    indeces = np.arange(len(classes))
    indeces_rf_rnf = np.arange(2)
    promise_exp_df = pd.read_csv('./datasets/PROMISE_exp.csv', sep=',', header=0, quotechar='"', doublequote=True)
    
    promise_exp_counts = get_counts(promise_exp_df, '_class_')
    
    promise_rf_rnf_counts = {'Funcionais': promise_exp_df[promise_exp_df['_class_'] == 'F'].shape[0],
                             'Não Funcionais': promise_exp_df[promise_exp_df['_class_'] != 'F'].shape[0]}
    
    graficoBarras(promise_exp_counts.values(), promise_exp_counts.keys(), indeces, "Distribuição entre as 12 classes de requisitos funcionais e não funcionais", 'graficoBarras12Classes.png')
    
    graficoPizzas(promise_exp_counts.values(), promise_exp_counts.keys(), "Porcetagem entre as 12 classes de requisitos funcionais e não funcionais", 'graficoPizza12Classes.png')
    
    graficoBarras(promise_rf_rnf_counts.values(), promise_rf_rnf_counts.keys(), indeces_rf_rnf, "Distribuição entre as classes de requisitos funcionais e não funcionais", 'graficoPizzaBarrasFuncionaisNaoFuncionais.png')
    
    graficoPizzas(promise_rf_rnf_counts.values(), promise_rf_rnf_counts.keys(), "Porcetagem entre as classes de requisitos funcionais e não funcionais", 'graficoPizzaClassesFuncionaisNaoFuncionais.png')

