"""Consultas e inferência do produto local; nenhuma alteração no experimento."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / 'artefatos_app'
FIELDS = ['empresa', 'origem', 'destino', 'mes', 'dia_semana', 'faixa_horaria_prevista']
MEASURES = ['linhas_brutas','elegiveis','atrasadas_30','canceladas','quarentena','realizadas_sem_horario','demais_excluidas','horarios_extremos']
MONTHS = ['Janeiro','Fevereiro','Março','Abril','Maio','Junho','Julho','Agosto','Setembro','Outubro','Novembro','Dezembro']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=1)
def load_data():
    manifest = read_json(DATA / 'manifesto_produto.json')
    for name,digest in manifest['arquivos'].items():
        actual=hashlib.sha256((DATA/name).read_bytes()).hexdigest()
        if actual!=digest: raise ValueError('Arquivo do produto divergente: '+name)
    return dict(cubo=pd.read_parquet(DATA/'cubo_indicadores.parquet'),
                predicoes=pd.read_parquet(DATA/'predicoes_teste.parquet'),
                rotas=pd.read_parquet(DATA/'rotas.parquet'),
                aeroportos=pd.read_parquet(DATA/'aeroportos.parquet'),
                empresas=pd.read_parquet(DATA/'empresas.parquet'),
                resultado=read_json(DATA/'resultado_sprint3.json'),manifesto=manifest)


@lru_cache(maxsize=1)
def load_predictor():
    load_data()
    return joblib.load(DATA/'encoder.joblib'),joblib.load(DATA/'modelo_preditivo.joblib')


def normalize_filters(filters=None):
    filters=filters or {}
    return {k:str(filters[k]) for k in ['mes_arquivo','origem','empresa'] if filters.get(k) and filters[k]!='Todos'}


def filter_frame(frame,filters=None):
    selected=frame
    for field,value in normalize_filters(filters).items(): selected=selected.loc[selected[field].astype(str).eq(value)]
    return selected.copy()


def summary(filters=None):
    selected=filter_frame(load_data()['cubo'],filters)
    values={m:int(selected[m].sum()) for m in MEASURES}
    values['taxa']=values['atrasadas_30']/values['elegiveis'] if values['elegiveis'] else None
    values['demais_exclusoes']=values['demais_excluidas']+values['realizadas_sem_horario']
    return values


def grouped(field,filters=None):
    frame=filter_frame(load_data()['cubo'],filters)
    result=frame.groupby(field,as_index=False)[MEASURES].sum()
    result['taxa']=result.atrasadas_30.div(result.elegiveis.where(result.elegiveis.ne(0)))
    return result


def risk_summary(filters=None):
    frame=filter_frame(load_data()['predicoes'],filters)
    values=dict(n=len(frame),positivos=int(frame.atrasado_30.sum()),priorizados=int(frame.priorizado_global_10.sum()),acertos=int(frame.acerto_global_10.sum()))
    values['recall']=values['acertos']/values['positivos'] if values['positivos'] else None
    values['precisao']=values['acertos']/values['priorizados'] if values['priorizados'] else None
    return values


def priorities(filters=None):
    frame=filter_frame(load_data()['predicoes'],filters)
    return frame.loc[frame.priorizado_global_10].sort_values(['escore','empresa','numero_voo','di','origem','destino','partida_prevista'],ascending=[False,True,True,True,True,True,True],kind='stable')


def predict_flight(empresa,origem,destino,partida_prevista):
    timestamp=pd.Timestamp(partida_prevista)
    if pd.isna(timestamp) or timestamp.tzinfo is not None: raise ValueError('Informe uma data válida em horário de Brasília, sem conversão de fuso.')
    codes=[str(v).strip().upper() for v in [empresa,origem,destino]]
    if not all(codes) or any(not c.isalpha() for c in codes): raise ValueError('Informe códigos ICAO válidos de empresa, origem e destino.')
    if codes[1]==codes[2]: raise ValueError('Origem e destino devem ser diferentes.')
    encoder,model=load_predictor()
    x=pd.DataFrame([[*codes,str(timestamp.month),str(timestamp.dayofweek),str(timestamp.hour//4)]],columns=FIELDS)
    score=float(model.predict_proba(encoder.transform(x))[0,1])
    unknown=[field for field,categories in zip(FIELDS,encoder.categories_) if x.iloc[0][field] not in categories]
    routes=load_data()['rotas']
    known=bool(((routes.empresa==codes[0])&(routes.origem==codes[1])&(routes.destino==codes[2])).any())
    return dict(empresa=codes[0],origem=codes[1],destino=codes[2],partida_prevista=timestamp.isoformat(),escore=score,
                atributos=x.iloc[0].to_dict(),categorias_nao_vistas=unknown,rota_observada=known,modelo=load_data()['resultado']['escolhido'],
                interpretacao='Escore experimental para ordenar risco; não representa probabilidade calibrada nem confirmação de atraso.')


def context_label(filters=None):
    values=normalize_filters(filters)
    return ' · '.join([values.get('mes_arquivo','2025 completo'),values.get('origem','todas as origens'),values.get('empresa','todas as empresas')])


def number(value): return f'{int(value):,}'.replace(',','.')
def percent(value): return '—' if value is None or pd.isna(value) else f'{100*value:.2f}%'.replace('.',',')
