"""Produto final local: indicadores, risco, inferência e assistente com IA."""
from datetime import datetime,date,time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import altair as alt
import pandas as pd
import streamlit as st
from produto_voos import load_data,summary,grouped,risk_summary,priorities,predict_flight,normalize_filters,context_label,number,percent,MONTHS
from assistente_voos import answer

st.set_page_config(page_title='Pontualidade em foco',page_icon='✈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
:root{font-family:"Segoe UI",Arial,sans-serif} .stApp{background:#f3f6f5;color:#163d35}
[data-testid="stSidebar"]{background:#123e35}[data-testid="stSidebar"] *{color:#f0faf5}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#163d35}
[data-testid="stSidebar"] button *{color:#163d35}[data-testid="stSidebar"] h2{font-size:1.35rem}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1440px}
header[data-testid="stHeader"]{background:transparent}[data-testid="stToolbar"]{visibility:hidden}
.hero{padding:26px 30px;background:#123e35;border-radius:16px;color:white;margin-bottom:22px}
.hero .eyebrow{font-size:.85rem;letter-spacing:1.7px;color:#9ee1bb;text-transform:uppercase}
.hero h1{font-size:2.5rem;line-height:1.1;color:white;padding:8px 0 6px;margin:0}
.hero p{font-size:1rem;margin:8px 0 0;color:#d3e8df}.section-note{font-size:1rem;color:#526c64}
[data-testid="stMetric"]{background:white;border:1px solid #d5e4dd;border-radius:12px;padding:17px 18px;min-height:132px}
[data-testid="stMetricValue"]{font-size:2rem;color:#123e35}[data-testid="stMetricLabel"]{font-size:1rem}
.stTabs [data-baseweb="tab-list"]{gap:22px;margin:14px 0 18px}.stTabs [data-baseweb="tab"]{font-size:1rem;height:48px}
.stTabs [aria-selected="true"]{color:#197447!important}.stTabs [data-baseweb="tab-highlight"]{background-color:#238456}
.stButton button,.stDownloadButton button{border-radius:8px;min-height:42px}.stCaption{font-size:.88rem}
[data-testid="stChatMessage"]{background:white;border:1px solid #dce8e2;border-radius:12px}
@media(max-width:700px){.hero{padding:20px}.hero h1{font-size:1.9rem}.block-container{padding:1rem}[data-testid="stMetricValue"]{font-size:1.7rem}}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def data():return load_data()


def reset_filters():
    for key in ['f_mes','f_origem','f_empresa']:st.session_state[key]='Todos'


def render_reply(reply):
    st.markdown(reply['texto'])
    if reply.get('tabela'):st.dataframe(pd.DataFrame(reply['tabela']),hide_index=True,width='stretch')
    st.caption('Recorte da resposta: '+reply.get('recorte',context_label(reply.get('filtros'))))
    with st.expander('Fonte da resposta'):
        st.write(reply['fonte'])
        st.caption('IA local identifica a intenção; os números são calculados nos arquivos do projeto.')


d=data();cube=d['cubo'];result=d['resultado']
with st.sidebar:
    st.markdown('### PONTUALIDADE\nConsulta & planejamento')
    st.caption('Histórico VRA / ANAC · 2025')
    st.divider()
    st.header('Seu recorte')
    month=st.selectbox('Mês',['Todos']+sorted(cube.mes_arquivo.unique()),key='f_mes')
    origin=st.selectbox('Origem',['Todos']+sorted(cube.origem.unique()),key='f_origem')
    company=st.selectbox('Empresa',['Todos']+sorted(cube.empresa.unique()),key='f_empresa')
    st.button('Limpar filtros',on_click=reset_filters,width='stretch',key='limpar')
    st.divider()
    st.markdown('**Como ler os números**')
    st.write('Atraso de partida maior que 30 minutos. Cancelamentos aparecem separados da taxa de atraso.')
    st.caption('Consulta histórica. Simulações são experimentais; o escore não é probabilidade calibrada.')

filters=normalize_filters(dict(mes_arquivo=month,origem=origin,empresa=company));total=summary(filters)
st.markdown('''<section class="hero"><div class="eyebrow">Projeto Aplicado · Engenharia de Dados e IA</div><h1>Pontualidade em foco</h1><p>Entenda os atrasos, explore o histórico e investigue o risco com contexto.</p></section>''',unsafe_allow_html=True)
st.caption('RECORTE ATIVO  ·  '+context_label(filters))
overview,risk,simulate,chat,about=st.tabs(['Visão geral','Risco histórico','Simular voo','Assistente IA','Sobre os dados'])

with overview:
    a,b,c,e=st.columns(4)
    a.metric('Taxa de atraso > 30 min',percent(total['taxa']));b.metric('Etapas elegíveis',number(total['elegiveis']));c.metric('Atrasos > 30 min',number(total['atrasadas_30']));e.metric('Cancelamentos',number(total['canceladas']))
    st.caption(f"Taxa = {number(total['atrasadas_30'])} atrasos ÷ {number(total['elegiveis'])} etapas realizadas elegíveis. Cancelamentos ficam fora desse denominador.")
    if not total['linhas_brutas']:st.info('Nenhum registro para esta combinação. A taxa é indefinida quando o denominador é zero.')
    left,right=st.columns([1.35,1],gap='large')
    with left:
        st.subheader('Pontualidade mês a mês')
        st.caption('Taxa recalculada a partir dos numeradores e denominadores do recorte.')
        monthly=grouped('mes_arquivo',filters)
        if not monthly.empty:
            chart=alt.Chart(monthly).mark_bar(color='#26885d',cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(x=alt.X('mes_arquivo:N',title='Mês de 2025',axis=alt.Axis(labelAngle=0,labelExpr="slice(datum.label,5,7)")),y=alt.Y('taxa:Q',title='Taxa de atraso',axis=alt.Axis(format='.0%')),tooltip=[alt.Tooltip('mes_arquivo:N',title='Mês'),alt.Tooltip('elegiveis:Q',title='Elegíveis',format=','),alt.Tooltip('atrasadas_30:Q',title='Atrasos',format=','),alt.Tooltip('taxa:Q',title='Taxa',format='.2%')]).properties(height=310)
            st.altair_chart(chart,use_container_width=True)
    with right:
        st.subheader('Origens com maior volume')
        origins=grouped('origem',filters).sort_values('elegiveis',ascending=False).head(8)
        display=origins[['origem','elegiveis','atrasadas_30','taxa']].rename(columns={'origem':'Origem','elegiveis':'Elegíveis','atrasadas_30':'Atrasos','taxa':'Taxa'})
        display['Taxa']=display['Taxa'].map(percent)
        st.dataframe(display,hide_index=True,width='stretch',height=333)
    with st.expander('Qualidade e reconciliação do recorte'):
        st.write(f"{number(total['linhas_brutas'])} linhas = {number(total['elegiveis'])} elegíveis + {number(total['canceladas'])} canceladas + {number(total['quarentena'])} em quarentena + {number(total['demais_exclusoes'])} demais exclusões.")
        st.write(f"Horários extremos sinalizados: {number(total['horarios_extremos'])}. Eles já integram as elegíveis; não devem ser somados novamente.")
    st.download_button('Baixar indicadores do recorte',monthly.to_csv(index=False).encode('utf-8-sig'),'indicadores_recorte.csv','text/csv',key='baixar_indicadores')

with risk:
    st.subheader('Uma lista de prioridades, com limites claros')
    st.write('Avaliação final: outubro a dezembro/2025. A lista global contém os 19.455 maiores escores do teste. Os filtros recortam essa lista fixa; não selecionam novamente 10% dentro do recorte.')
    rs=risk_summary(filters);a,b,c,e=st.columns(4)
    a.metric('Recall da lista no recorte',percent(rs['recall']));b.metric('Etapas do teste no recorte',number(rs['n']));c.metric('Etapas prioritárias',number(rs['priorizados']));e.metric('Precisão da lista no recorte',percent(rs['precisao']))
    st.caption(f"{number(rs['acertos'])} atrasos encontrados em {number(rs['priorizados'])} prioridades, de {number(rs['positivos'])} atrasos reais no recorte.")
    if not rs['n']:st.info('Não há etapas do teste final neste recorte. Meses fora de outubro a dezembro permanecem disponíveis em Visão geral.')
    st.subheader('Comparação dos modelos no teste completo')
    comparison=pd.DataFrame([{'Candidato':r['candidato']+(' · escolhido' if r['candidato']==result['escolhido'] else ''),'Recall@10%':percent(r['recall_10']),'Precisão@10%':percent(r['precisao_10']),'Average Precision':round(r['average_precision'],4)} for r in result['teste']])
    st.dataframe(comparison,hide_index=True,width='stretch')
    st.caption('Tabela fixa do experimento. M2_D6 foi escolhido pela validação, antes de avaliar o teste. Recall@10% no teste: 16,21%; meta exploratória: 20%.')
    st.subheader('Etapas priorizadas neste recorte')
    priority=priorities(filters)
    st.dataframe(priority[['empresa','numero_voo','origem','destino','partida_prevista','escore','atrasado_30']].head(100),hide_index=True,width='stretch')
    st.download_button('Baixar lista prioritária do recorte',priority.to_csv(index=False).encode('utf-8-sig'),'prioridades_recorte.csv','text/csv',key='baixar_prioridades')

with simulate:
    st.subheader('Explore uma etapa de voo')
    st.write('Informe a programação para executar o modelo treinado. A simulação usa empresa, origem, destino, mês, dia da semana e faixa horária; não solicita o resultado real do voo.')
    routes=d['rotas'];companies=sorted(routes.empresa.unique());default_company=companies.index('TAM') if 'TAM' in companies else 0
    a,b,c=st.columns(3)
    with a:sim_company=st.selectbox('Empresa do voo',companies,index=default_company,key='sim_empresa')
    origins=sorted(routes.loc[routes.empresa.eq(sim_company)].origem.unique());default_origin=origins.index('SBGR') if 'SBGR' in origins else 0
    with b:sim_origin=st.selectbox('Origem do voo',origins,index=default_origin,key='sim_origem')
    destinations=sorted(routes.loc[routes.empresa.eq(sim_company)&routes.origem.eq(sim_origin)].destino.unique())
    with c:sim_destination=st.selectbox('Destino do voo',destinations,key='sim_destino')
    with st.form('simulacao'):
        a,b=st.columns(2)
        with a:sim_date=st.date_input('Data prevista',value=date(2025,11,15),min_value=date(2025,1,1),max_value=date(2030,12,31),key='sim_data')
        with b:sim_time=st.time_input('Horário previsto — Brasília',value=time(16,0),key='sim_hora')
        submitted=st.form_submit_button('Estimar risco',type='primary',width='stretch')
    if submitted:
        st.session_state['ultima_simulacao']=predict_flight(sim_company,sim_origin,sim_destination,datetime.combine(sim_date,sim_time))
    if st.session_state.get('ultima_simulacao'):
        prediction=st.session_state['ultima_simulacao']
        st.success('Inferência executada com o modelo treinado.')
        a,b=st.columns([1,2])
        a.metric('Escore experimental',f"{prediction['escore']:.4f}".replace('.',','))
        b.write(f"**{prediction['empresa']} · {prediction['origem']} → {prediction['destino']}**  \nPartida prevista: {pd.Timestamp(prediction['partida_prevista']).strftime('%d/%m/%Y às %H:%M')} · Brasília")
        st.write(prediction['interpretacao'])
        if prediction['categorias_nao_vistas']:st.caption('Tratamento de categorias novas aplicado conforme o protocolo do modelo. Campos: '+', '.join(prediction['categorias_nao_vistas'])+'.')
        st.caption('Modelo M2_D6. Validação retrospectiva em 2025; sem garantia de disponibilidade operacional anterior à partida.')

with chat:
    st.subheader('Converse com os dados')
    st.write('Pergunte sobre indicadores, meses, origens, cancelamentos, qualidade, o experimento ou sua última simulação. O assistente considera o recorte lateral e identifica meses ou códigos informados na pergunta.')
    st.caption('Assistente com IA local: processamento de linguagem natural e consultas verificáveis. Sem serviço externo ou modelo generativo.')
    if 'mensagens' not in st.session_state:st.session_state['mensagens']=[]
    suggestions=st.columns(3)
    for col,label in zip(suggestions,['Resumo do recorte','Compare os meses','Qual modelo foi escolhido?']):
        if col.button(label,width='stretch',key='perg_'+label):st.session_state['pergunta_sugerida']=label
    if st.button('Limpar conversa',key='limpar_chat'):st.session_state['mensagens']=[]
    for message in st.session_state['mensagens']:
        with st.chat_message('user'):st.write(message['pergunta'])
        with st.chat_message('assistant'):render_reply(message['resposta'])
    prompt=st.chat_input('Ex.: Qual a taxa em agosto na SBGR com a TAM?',key='pergunta_assistente',max_chars=1500)
    question=st.session_state.pop('pergunta_sugerida',None) or prompt
    if question:
        previous=st.session_state['mensagens'][-1]['resposta'] if st.session_state['mensagens'] else None
        reply=answer(question,filters,previous,st.session_state.get('ultima_simulacao'))
        st.session_state['mensagens'].append(dict(pergunta=question,resposta=reply));st.session_state['mensagens']=st.session_state['mensagens'][-20:]
        with st.chat_message('user'):st.write(question)
        with st.chat_message('assistant'):render_reply(reply)

with about:
    st.subheader('Origem e significado dos dados')
    st.write('Fonte: Voo Regular Ativo — ANAC/SIROS, arquivos de janeiro a dezembro/2025. Cada linha da base analítica conserva a versão e o registro de origem. A unidade é uma etapa entre origem e destino.')
    st.write('Recorte: linha N, DI 0, assentos > 0 e partida prevista no mês nominal. Atraso é a diferença estritamente maior que 30 minutos entre partida real e prevista; cancelamentos não entram no denominador.')
    st.subheader('Dois componentes de IA')
    st.write('O classificador de risco usa a programação para produzir escores. O assistente interpreta intenções em português com TF-IDF e regressão logística e responde a partir das consultas locais. Não há geração livre de fatos, acesso à internet ou envio de dados a uma API.')
    st.subheader('Alcance da avaliação')
    st.write('A avaliação é histórica e técnica. A programação consolidada não demonstra o momento em que cada informação ficou disponível. Escores não são probabilidades calibradas; as análises não identificam causas nem demonstram redução de atrasos.')
    st.write('Os testes realizados verificam consultas, inferência, linguagem e interface. Não houve estudo com participantes, medição de satisfação ou comparação de tempo humano de execução de tarefas.')
    st.link_button('Fonte oficial — ANAC/SIROS','https://siros.anac.gov.br/siros/registros/diversos/vra/2025/')

st.caption('Marco Antonio Luiz Giampietro Junior · Projeto Aplicado · XP Educação · Entrega final de 23/11/2026')
