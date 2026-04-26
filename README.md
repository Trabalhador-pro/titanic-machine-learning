# Titanic - Machine Learning

Este projeto tem como objetivo prever quais passageiros sobreviveram ao naufrágio do Titanic, utilizando técnicas de Machine Learning.

## Objetivo

Criar um modelo de classificação capaz de prever a variável `Survived`, indicando se o passageiro sobreviveu ou não.

## Base de dados

Foram utilizadas duas bases:

- `train.csv`: base com os dados dos passageiros e a variável alvo `Survived`.
- `test.csv`: base sem a variável alvo, usada para gerar as previsões finais.

## Etapas do projeto

1. Carregamento dos dados
2. Análise inicial das variáveis
3. Verificação de valores ausentes
4. Remoção de colunas pouco úteis para a primeira versão
5. Criação de pipeline de pré-processamento
6. Treinamento de modelo com Árvore de Decisão
7. Avaliação com acurácia e precisão
8. Geração do arquivo final de submissão

## Tratamento dos dados

Foram removidas as colunas:

- `Cabin`: alto percentual de valores ausentes
- `Ticket`: alta cardinalidade e pouca utilidade direta nesta primeira versão
- `Name`: alta cardinalidade e não utilizada nesta abordagem inicial
- `PassengerId`: usado apenas como identificador final

As variáveis numéricas receberam imputação pela média.  
As variáveis categóricas receberam imputação pela moda e codificação com `OrdinalEncoder`.

## Modelo utilizado

O modelo escolhido foi uma Árvore de Decisão:
```python
DecisionTreeClassifier(max_depth=3, random_state=42)
```
## Resultado

O modelo alcançou aproximadamente **77% de acurácia**.

Como este foi meu primeiro projeto de Machine Learning, o principal objetivo não foi apenas buscar a maior performance possível, mas compreender toda a estrutura de desenvolvimento de um modelo, desde a análise inicial dos dados, tratamento de valores ausentes, criação de pipelines, treinamento, avaliação e geração das previsões finais.

Mesmo entendendo que ainda existem muitas melhorias possíveis e que este resultado pode ser bastante otimizado, considero este projeto uma etapa extremamente importante no meu processo de aprendizado e desenvolvimento na área de Ciência de Dados.

Esta foi minha primeira versão do projeto, construída com o objetivo de consolidar a base necessária para projetos mais robustos e profissionais no futuro.

Pretendo revisitar este projeto futuramente, aplicar novas técnicas, testar modelos mais avançados, melhorar a engenharia de atributos e publicar uma nova versão com melhorias mais consistentes e uma performance superior.
