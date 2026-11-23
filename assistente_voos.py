"""Assistente analítico: NLU supervisionada local e respostas de consultas reais.

O classificador identifica intenção; as contagens vêm das consultas, nunca
dos exemplos linguísticos. Não é um LLM e não gera SQL ou fatos livres.
"""
from __future__ import annotations
from functools import lru_cache
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.linear_model import LogisticRegression
from produto_voos import DATA,load_data,summary,grouped,risk_summary,predict_flight,normalize_filters,context_label,number,percent,MONTHS

INTENTS = {
 'resumo':['resuma os dados','qual a taxa de atraso','quantos voos atrasaram','mostre os indicadores','me dê um panorama','resumo do recorte','qual a pontualidade','como estão os atrasos','quantas etapas são elegíveis','qual o percentual de atraso','quantos atrasos temos nesse filtro','volume e taxa de atraso','quero uma visão geral dos voos','qual o total de voos realizados','quero saber os números gerais'],
 'origens':['quais aeroportos têm mais atrasos','compare as origens','mostre os aeroportos','qual origem tem maior taxa','ranking dos aeroportos por atraso','quais aeroportos têm mais voos','compare aeroportos de origem','aeroportos com maior volume','ordene origens pela taxa','onde há maior percentual de atraso','quais origens estão na liderança','qual aeroporto concentra atrasos','analisar aeroportos','pontualidade por aeroporto','mostrar ranking por origem'],
 'meses':['compare os meses','evolução mensal dos atrasos','qual mês tem maior taxa','tendência por mês','como evoluiu a pontualidade','melhor e pior mês','mostre a série temporal','análise mensal de 2025','compare janeiro com dezembro','taxa em cada mês','há variação entre os meses','em qual mês houve mais atrasos','ranking mensal','qual mês apresentou maior atraso','histórico mês a mês'],
 'cancelamentos':['quantos voos foram cancelados','mostre cancelamentos','taxa de cancelamento','os cancelados entram na taxa','cancelamentos no recorte','quero saber sobre voos cancelados','quantas etapas canceladas','exibir cancelados','há cancelamentos nos dados','total de cancelamentos','qual a quantidade de voos cancelados','cancelados são pontuais','compare canceladas e realizadas','entender os cancelamentos','veja quantos cancelaram'],
 'qualidade':['qual a qualidade dos dados','existem duplicidades','quantos registros em quarentena','há horários extremos','quais as exclusões','reconciliação das linhas','os dados estão completos','há dados faltantes','registros rejeitados','problemas de qualidade','quantas linhas foram excluídas','como as linhas foram classificadas','duplicados e horários inválidos','quantos registros foram sinalizados','veja os dados excluídos'],
 'modelo':['qual modelo foi escolhido','qual foi o desempenho da IA','compare os modelos','a meta foi atingida','qual o recall no teste','métricas do experimento','resultados da árvore','qual a precisão do modelo','quanto deu a average precision','a IA teve um bom resultado','qual classificador venceu','como o modelo foi selecionado','desempenho preditivo','qual resultado final da previsão','relembre o experimento'],
 'ranking':['mostre a lista de risco','quantos voos foram priorizados','qual o recall da lista neste recorte','lista prioritária do teste','quais etapas são prioritárias','qual o ranking de risco histórico','como funciona a seleção dos dez por cento','atrasos encontrados na lista','mostre o grupo de maior risco','quantas etapas estão na lista global','precisão da priorização','prioridades do modelo','quem entrou nos 10%','quero ver o risco histórico','dados do teste priorizados'],
 'escore':['o que significa esse escore','o escore é probabilidade','explique o risco do voo simulado','interprete minha simulação','como prever um novo voo','como simular uma etapa','explique o resultado da previsão','o voo simulado vai atrasar','resultado da simulação','o que significa o valor do risco','posso interpretar como chance de atraso','esse percentual é calibrado','é uma certeza de atraso','explique a inferência do modelo','quero estimar um voo'],
 'fonte':['de onde vêm os dados','qual a fonte dos registros','como rastrear os arquivos','quem publica os voos','origem da base','de onde saíram os números','onde encontro os dados da ANAC','qual o período analisado','os dados são oficiais','fonte e período','metadados dos voos','qual é a base usada','rastreabilidade dos arquivos','a base tem todos os meses','que ano está sendo consultado'],
 'definicoes':['o que é atraso','qual o denominador da taxa','explique os indicadores','o que é etapa elegível','qual a fórmula da taxa de atraso','por que usar trinta minutos','qual a definição de pontualidade','quais voos entram no cálculo','atraso inclui exatamente trinta minutos','o que significa realizada elegível','quais são as regras de inclusão','como calcular a taxa','entenda numerador e denominador','como vocês definem atraso','que unidade de análise é usada'],
 'limites':['qual a causa dos atrasos','por que os voos atrasam','a chuva causou os atrasos','quais as limitações do projeto','a IA reduz atrasos','posso usar em tempo real','houve validação com usuários','quem foi entrevistado','qual o ganho operacional','o modelo garante o resultado','há prova de redução de custos','esse resultado demonstra causalidade','como foi a pesquisa com pessoas','posso aplicar em operação real','isso prova que o aeroporto causa atraso'],
 'ajuda':['como usar o assistente','o que posso perguntar','quais perguntas você responde','me ajude com o painel','como usar os filtros','como funciona esta aplicação','olá','oi assistente','preciso de ajuda','que análises estão disponíveis','quais suas funções','como navegar nas telas','o que você consegue fazer','começar uma análise','orientações de uso']
}

# Perguntas reservadas, escritas separadamente dos exemplos de ajuste.
HOLDOUT = [
 ('Pode resumir a quantidade de atrasos e a taxa deste recorte?','resumo'),('Qual a porcentagem de voos atrasados?','resumo'),
 ('Liste as origens com maior proporção de atrasos.','origens'),('Qual aeroporto teve a maior taxa?','origens'),
 ('Mostre uma comparação entre os meses do ano.','meses'),('Em que mês a pontualidade foi pior?','meses'),
 ('Qual o número de cancelamentos?','cancelamentos'),('Voos cancelados fazem parte do denominador?','cancelamentos'),
 ('Quantas duplicidades foram para quarentena?','qualidade'),('Quais problemas de qualidade foram identificados?','qualidade'),
 ('Qual foi o modelo vencedor e o recall no teste?','modelo'),('Mostre os resultados dos classificadores.','modelo'),
 ('Qual a quantidade de etapas priorizadas no teste?','ranking'),('Como está a precisão da lista prioritária?','ranking'),
 ('Esse escore pode ser lido como probabilidade?','escore'),('Como interpretar o voo que simulei?','escore'),
 ('Qual conjunto oficial alimenta o painel?','fonte'),('De qual ano são esses registros?','fonte'),
 ('Como é definido o atraso de partida?','definicoes'),('Que população compõe o denominador?','definicoes'),
 ('É possível concluir que o clima causou o atraso?','limites'),('Vocês entrevistaram analistas?','limites'),
 ('Quais perguntas posso fazer aqui?','ajuda'),('Pode me orientar a usar o painel?','ajuda')
]


def train_assistant():
    questions=[q for items in INTENTS.values() for q in items]
    labels=[intent for intent,items in INTENTS.items() for _ in items]
    model=Pipeline([
        ('texto',FeatureUnion([
            ('palavras',TfidfVectorizer(strip_accents='unicode',lowercase=True,ngram_range=(1,2),sublinear_tf=True)),
            ('caracteres',TfidfVectorizer(strip_accents='unicode',lowercase=True,analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True))])),
        ('intencao',LogisticRegression(C=8,class_weight='balanced',max_iter=2000,random_state=42))])
    model.fit(questions,labels)
    artifact=dict(modelo=model,perguntas=questions,intencoes=labels,vetores=model.named_steps['texto'].transform(questions),
                  tecnica='TF-IDF de palavras e caracteres + regressão logística; consultas determinísticas e respostas fundamentadas.',
                  exemplos='Exemplos linguísticos sintéticos escritos para esta aplicação, sem participantes ou conversas reais.')
    joblib.dump(artifact,DATA/'assistente_nlu.joblib',compress=3)
    return artifact


@lru_cache(maxsize=1)
def load_assistant():
    load_data()
    return joblib.load(DATA/'assistente_nlu.joblib')


def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.lower()) if not unicodedata.combining(c))


def parse_filters(question,base=None):
    filters=normalize_filters(base)
    text=plain(question)
    known=load_data()
    for field,codes in [('origem',set(known['cubo'].origem)),('empresa',set(known['cubo'].empresa))]:
        for token in re.findall(r'\b[A-Za-z]{3,4}\b',question):
            if token.upper() in codes:filters[field]=token.upper();break
    month_match=re.search(r'\b2025[-/](0[1-9]|1[0-2])\b',text)
    if month_match:filters['mes_arquivo']='2025-'+month_match.group(1)
    else:
        for i,name in enumerate(MONTHS,1):
            if re.search(r'\b'+plain(name)+r'\b',text):filters['mes_arquivo']=f'2025-{i:02d}';break
    if any(v in text for v in ['ano inteiro','todos os meses','ano completo']):filters.pop('mes_arquivo',None)
    return filters


def classify(question):
    artifact=load_assistant(); model=artifact['modelo']
    vector=model.named_steps['texto'].transform([question])
    similarities=(vector@artifact['vetores'].T).toarray()[0]/2
    similarity=float(similarities.max())
    intent=str(model.predict([question])[0])
    return intent,similarity


def answer(question,filters=None,previous=None,last_prediction=None):
    question=str(question).strip()[:1500]
    previous=previous or {}
    intent,similarity=classify(question)
    normalized=plain(question)
    follow_up=bool(re.match(r'^(e |e\s+em |agora |nesse |neste |somente |apenas )',normalized)) and previous.get('intencao')
    selected=parse_filters(question,previous.get('filtros',filters) if follow_up else filters)
    if follow_up:intent=previous['intencao']
    # Guardas explícitas de domínio: o classificador não é autorizado a
    # atribuir causas ou afirmar pesquisa com pessoas por aproximação textual.
    if re.search(r'\b(causa|causou|causalidade|clima|chuva|entrevista\w*|participante\w*|satisfacao)\b',normalized):intent='limites'
    years=re.findall(r'\b20\d{2}\b',question)
    if years and any(year!='2025' for year in years) and intent!='escore':
        return dict(texto='A base histórica desta aplicação cobre somente 2025. Não há indicadores observados para o ano solicitado. Use Simular voo para executar uma estimativa experimental, sem confundi-la com resultado histórico.',intencao='fora_escopo',aderencia=similarity,filtros=selected,tabela=[],fonte='Cobertura temporal do manifesto do produto',titulo='')
    compared_months={f'2025-{i:02d}' for i,name in enumerate(MONTHS,1) if re.search(r'\b'+plain(name)+r'\b',normalized)}
    compared_months.update(re.findall(r'\b2025-(?:0[1-9]|1[0-2])\b',normalized))
    if intent=='meses' and len(compared_months)>1:selected.pop('mes_arquivo',None)
    forbidden=any(s in normalized for s in ['invente','ignore as regras','ignore instrucoes','finja','apague','delete','senha','api key'])
    if forbidden or (similarity<.13 and not follow_up):
        return dict(texto='Posso analisar pontualidade, cancelamentos, qualidade, ranking de risco e resultados deste projeto. Reformule a pergunta com um desses temas; não produzo números sem uma consulta à base.',intencao='fora_escopo',aderencia=similarity,filtros=selected,tabela=[],fonte='Escopo analítico local do projeto',titulo='')
    d=load_data(); total=summary(selected); result=d['resultado']; chosen=result['resultado_escolhido']
    rows=[]; title=''; source='cubo_indicadores.parquet — VRA/ANAC, 2025'
    if intent=='resumo':
        text=f"Neste recorte há **{number(total['elegiveis'])} etapas elegíveis**, **{number(total['atrasadas_30'])} atrasos > 30 minutos** e taxa de **{percent(total['taxa'])}**. A taxa usa apenas realizadas com horários válidos. Cancelamentos: {number(total['canceladas'])}, contabilizados separadamente."
        if not total['linhas_brutas']:text='Não há registros para esse recorte. A taxa é indefinida quando o denominador é zero.'
    elif intent in ['origens','meses']:
        field='origem' if intent=='origens' else 'mes_arquivo'
        frame=grouped(field,selected); frame=frame.loc[frame.elegiveis.gt(0)]
        if intent=='meses' and len(compared_months)>1:frame=frame.loc[frame.mes_arquivo.isin(compared_months)]
        order='elegiveis' if any(v in normalized for v in ['volume','mais voos','mais etapas']) else 'atrasadas_30' if any(v in normalized for v in ['mais atrasos','quantidade de atrasos']) else 'taxa'
        frame=frame.sort_values([order,field],ascending=[False,True],kind='stable')
        if frame.empty:text='Não há etapas elegíveis para comparar neste recorte.'
        else:
            first=frame.iloc[0]
            lead=f"A maior taxa observada é de **{percent(first.taxa)}**" if order=='taxa' else f"O maior volume é de **{number(first.elegiveis)} etapas elegíveis**" if order=='elegiveis' else f"A maior contagem é de **{number(first.atrasadas_30)} atrasos**"
            text=f"{lead}, em **{first[field]}**, neste recorte. O grupo tem {number(first.elegiveis)} etapas elegíveis e taxa de {percent(first.taxa)}. A comparação é descritiva; volumes diferentes exigem cuidado na interpretação."
            rows=[{('Origem' if field=='origem' else 'Mês'):r[field],'Elegíveis':int(r.elegiveis),'Atrasos':int(r.atrasadas_30),'Taxa':percent(r.taxa)} for _,r in frame.head(12).iterrows()]
            title='Comparação por '+('origem' if intent=='origens' else 'mês')
    elif intent=='cancelamentos':
        text=f"Há **{number(total['canceladas'])} cancelamentos** neste recorte. Eles ficam fora do denominador da taxa de atraso, que contém {number(total['elegiveis'])} etapas realizadas elegíveis. Cancelamento não é classificado como pontualidade."
    elif intent=='qualidade':
        text=f"O recorte reúne {number(total['linhas_brutas'])} linhas: {number(total['elegiveis'])} elegíveis + {number(total['canceladas'])} canceladas + {number(total['quarentena'])} em quarentena + {number(total['demais_exclusoes'])} demais exclusões. Há {number(total['horarios_extremos'])} horários extremos sinalizados; eles já integram as elegíveis e não são somados novamente."
    elif intent=='modelo':
        text=f"O escolhido pela validação foi **{result['escolhido']}**, árvore de profundidade 6. No teste completo, o recall@10% foi **{percent(chosen['recall_10'])}**, precisão@10% **{percent(chosen['precisao_10'])}** e Average Precision **{chosen['average_precision']:.4f}**. A meta exploratória de 20% não foi atingida. A escolha foi congelada antes do teste, sem substituição posterior. Os filtros do painel não alteram essa comparação fixa."
        rows=[{'Candidato':r['candidato'],'Recall@10%':percent(r['recall_10']),'Precisão@10%':percent(r['precisao_10']),'AP':round(r['average_precision'],4)} for r in result['teste']];title='Experimento — teste completo';source='resultado_sprint3.json; seleção congelada antes do teste'
    elif intent=='ranking':
        risk=risk_summary(selected)
        text=f"No recorte do teste há **{number(risk['n'])} etapas**, com {number(risk['positivos'])} atrasos. A lista global fixa contém **{number(risk['priorizados'])}** dessas etapas e encontrou **{number(risk['acertos'])}** atrasos: recall de **{percent(risk['recall'])}** e precisão de **{percent(risk['precisao'])}** neste recorte. Filtrar não refaz os 10%: a lista foi selecionada uma vez no teste completo."
        if not risk['n']:text='Este recorte não contém etapas do teste final, que cobre outubro a dezembro/2025 após purga. Os indicadores históricos anuais continuam disponíveis em Visão geral.'
        source='predicoes_teste.parquet — lista global fixa de 19.455 etapas'
    elif intent=='escore':
        if last_prediction:
            text=f"A simulação **{last_prediction['empresa']} · {last_prediction['origem']} → {last_prediction['destino']}**, partida {last_prediction['partida_prevista']}, produziu escore **{last_prediction['escore']:.4f}** no modelo {last_prediction['modelo']}. Ele serve para ordenação experimental: **não é uma probabilidade calibrada nem garantia de atraso**. A avaliação do modelo é retrospectiva."
        else:text='Use a aba **Simular voo** para informar empresa, origem, destino e partida prevista. O modelo executa a inferência com esses atributos da programação. O escore resultante ordena risco experimental; não deve ser interpretado como probabilidade calibrada ou certeza de atraso.'
        source='modelo_preditivo.joblib; encoder.joblib; resultado_sprint3.json'
    elif intent=='fonte':
        text='A fonte é o **Voo Regular Ativo (VRA), ANAC/SIROS**, de janeiro a dezembro de 2025. Os 12 arquivos foram preservados com URL, data, tamanho e SHA-256. A base analítica classifica 1.005.807 linhas; 769.740 são etapas elegíveis. O conjunto é histórico consolidado, não uma captura de programação em tempo real.'
        source='manifesto_produto.json; manifestos de origem das Sprints 1 e 2'
    elif intent=='definicoes':
        text='Atraso é a diferença entre partida real e prevista **estritamente maior que 30 minutos**. Taxa = atrasos ÷ etapas realizadas elegíveis com horários válidos. Exatamente 30 minutos não entra no numerador. A unidade é uma etapa entre origem e destino; cancelamentos e exclusões têm contagens próprias. O recorte usa linha N, DI 0, assentos > 0 e partida prevista no mês do arquivo, com referência de Brasília.'
        source='Contrato de dados e regras preservados das Sprints 1 e 2'
    elif intent=='limites':
        text='Os registros permitem descrever atrasos e avaliar um ranking retrospectivo. **Não permitem concluir a causa dos atrasos**, medir redução operacional ou garantir antecipação real. A programação consolidada não possui timestamp de disponibilidade anterior ao voo. A avaliação realizada foi técnica e automatizada; não houve entrevistas, pesquisa com participantes ou medição de satisfação e ganho de tempo humano.'
        source='Escopo, protocolo preditivo e condições efetivas da avaliação'
    else:
        text='Posso resumir o recorte, comparar meses e origens, consultar cancelamentos e qualidade, explicar o modelo, a lista de risco e o escore de uma simulação. Experimente: **“Qual a taxa de atraso em agosto na SBGR com a TAM?”**. Os filtros do histórico são usados por padrão; um mês ou código informado na pergunta define o recorte da resposta. Uma continuação como “e em novembro?” mantém a intenção e o contexto da pergunta anterior.'
        source='Assistente local: classificação supervisionada de intenções e consultas aos dados'
    return dict(texto=text,intencao=intent,aderencia=similarity,filtros=selected,tabela=rows,fonte=source,titulo=title,recorte=context_label(selected))
