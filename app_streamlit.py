"""Produto final local: indicadores, risco, inferência e assistente com IA."""
from datetime import datetime,date,time
from html import escape
from numbers import Integral,Real
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import altair as alt
import pandas as pd
import streamlit as st
from produto_voos import load_data,summary,grouped,risk_summary,priorities,predict_flight,normalize_filters,context_label,number,percent,MONTHS
from assistente_voos import answer

st.set_page_config(page_title='Pontualidade em foco',page_icon='✈',layout='wide',initial_sidebar_state='auto')
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
.st-key-origins_mobile,.st-key-models_mobile,.st-key-priorities_mobile{display:none}
.mobile-chat-table{display:none}
.table-card{width:100%;border:1px solid #d5e4dd;background:white;border-radius:12px;overflow:hidden}
.compact-table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:.875rem}
.compact-table th,.compact-table td{padding:11px 7px;text-align:right;border-bottom:1px solid #e4ece7;overflow-wrap:anywhere;vertical-align:middle}
.compact-table th{background:#e8f2ec;color:#285044;font-weight:600;font-size:.875rem;line-height:1.3}
.compact-table th:first-child,.compact-table td:first-child{text-align:left;width:29%}
.compact-table tr:last-child td{border-bottom:0}.compact-table tbody tr:nth-child(even){background:#f7faf8}
.priority-card{background:white;border:1px solid #d5e4dd;border-radius:12px;padding:14px;margin-bottom:10px}
.priority-card .route{display:flex;justify-content:space-between;gap:12px;font-size:1rem;color:#164b38}
.priority-card p{font-size:.875rem!important;color:#526c64;margin:6px 0 0;line-height:1.45}
.stButton button,.stDownloadButton button{min-height:44px}.stButton button p,.stDownloadButton button p{white-space:normal}
@media(max-width:700px){
 .block-container{padding:3.3rem 1rem 1.5rem!important}
 .hero{padding:17px 18px;border-radius:12px;margin-bottom:12px}.hero .eyebrow{display:none}
 .hero h1{font-size:clamp(1.6rem,6.6vw,1.9rem);line-height:1.15;padding:0 0 4px;overflow-wrap:break-word}
 .hero p{font-size:1rem;line-height:1.45;margin-top:6px}
 [data-testid="stMain"] h2,[data-testid="stMain"] h3{font-size:1.35rem;line-height:1.3}
 [data-testid="stMain"] [data-testid="stCaptionContainer"]{font-size:.875rem;line-height:1.45;overflow-wrap:anywhere}
 .st-key-overview_metrics [data-testid="stHorizontalBlock"],.st-key-risk_metrics [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px!important}
 .st-key-overview_metrics [data-testid="stColumn"],.st-key-risk_metrics [data-testid="stColumn"]{width:100%!important;min-width:0!important;flex-basis:0!important}
 [data-testid="stMetric"]{padding:12px 11px;min-height:108px;height:100%;border-radius:10px}
 [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{font-size:.875rem!important;line-height:1.35;white-space:normal}
 [data-testid="stMetricLabel"]{min-height:38px}[data-testid="stMetricValue"]{font-size:1.5rem!important;line-height:1.25}
 .stTabs [data-baseweb="tab-list"]{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px!important;overflow:visible!important;height:auto!important;margin:0 0 14px;padding:0}
 .stTabs [data-baseweb="tab"]{width:100%;min-width:0;height:48px!important;min-height:44px;white-space:normal!important;padding:7px 5px;border:1px solid #d5e4dd;border-radius:8px;background:white;justify-content:center}
 .stTabs [data-baseweb="tab"] p{font-size:.875rem!important;line-height:1.2;white-space:normal!important;text-align:center}
 .stTabs [aria-selected="true"]{background:#dceee2!important;border-color:#88b89d!important;color:#15583a!important;font-weight:600}
 .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
 .st-key-origins_desktop,.st-key-models_desktop,.st-key-priorities_desktop{display:none}
 .st-key-origins_mobile,.st-key-models_mobile,.st-key-priorities_mobile{display:block}
 [data-testid="stExpander"] summary{min-height:44px}
 [data-testid="stChatMessage"]{padding:12px 10px;gap:8px;min-width:0}
 [data-testid="stChatMessage"] p{overflow-wrap:anywhere;font-size:1rem;line-height:1.5}
 [data-testid="stChatMessage"] [data-testid="stDataFrame"]{display:none}
 .mobile-chat-table{display:block}
 [data-testid="stChatInput"]{min-width:0}[data-testid="stChatInput"] textarea{font-size:1rem!important}
}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def data():return load_data()


def reset_filters():
    for key in ['f_mes','f_origem','f_empresa']:st.session_state[key]='Todos'


def render_reply(reply):
    st.markdown(reply['texto'])
    if reply.get('tabela'):
        frame=pd.DataFrame(reply['tabela'])
        st.dataframe(frame,hide_index=True,width='stretch')
        mobile_table(frame,'mobile-chat-table')
    st.caption('Recorte da resposta: '+reply.get('recorte',context_label(reply.get('filtros'))))
    with st.expander('Fonte da resposta'):
        st.write(reply['fonte'])
        st.caption('IA local identifica a intenção; os números são calculados nos arquivos do projeto.')


def mobile_table(frame,css_class=''):
    def formatted(value):
        if isinstance(value,Integral):return number(value)
        if isinstance(value,Real):return '—' if pd.isna(value) else f'{value:.4f}'.replace('.',',')
        return str(value)
    headers=''.join('<th>'+escape(str(c)).replace('@','<br>@')+'</th>' for c in frame.columns)
    rows=''.join('<tr>'+''.join('<td>'+escape(formatted(v))+'</td>' for v in row)+'</tr>' for row in frame.itertuples(index=False,name=None))
    if not rows:rows='<tr><td colspan="'+str(len(frame.columns))+'">Sem dados neste recorte.</td></tr>'
    st.markdown('<div class="table-card '+css_class+'"><table class="compact-table"><thead><tr>'+headers+'</tr></thead><tbody>'+rows+'</tbody></table></div>',unsafe_allow_html=True)


d=data();cube=d['cubo'];result=d['resultado']
st.markdown('''<section class="hero"><div class="eyebrow">Projeto Aplicado · Engenharia de Dados e IA</div><h1>Pontualidade em foco</h1><p>Indicadores, simulação e assistente para investigar atrasos.</p></section>''',unsafe_allow_html=True)
with st.expander('Filtrar histórico',expanded=False):
    filter_month,filter_origin,filter_company,filter_reset=st.columns([1,1,1,.8],vertical_alignment='bottom')
    with filter_month:month=st.selectbox('Mês',['Todos']+sorted(cube.mes_arquivo.unique()),key='f_mes')
    with filter_origin:origin=st.selectbox('Origem',['Todos']+sorted(cube.origem.unique()),key='f_origem')
    with filter_company:company=st.selectbox('Empresa',['Todos']+sorted(cube.empresa.unique()),key='f_empresa')
    with filter_reset:st.button('Limpar filtros',on_click=reset_filters,width='stretch',key='limpar')

filters=normalize_filters(dict(mes_arquivo=month,origem=origin,empresa=company));total=summary(filters)
st.caption('RECORTE ATIVO  ·  '+context_label(filters))
overview,risk,simulate,chat,about=st.tabs(['Visão geral','Risco histórico','Simular voo','Assistente IA','Sobre os dados'])

with overview:
    with st.container(key='overview_metrics'):
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
        with st.container(key='origins_desktop'):st.dataframe(display,hide_index=True,width='stretch',height=333)
        compact=display.copy();compact['Elegíveis']=compact['Elegíveis'].map(number);compact['Atrasos']=compact['Atrasos'].map(number)
        with st.container(key='origins_mobile'):mobile_table(compact)
    with st.expander('Qualidade e reconciliação do recorte'):
        st.write(f"{number(total['linhas_brutas'])} linhas = {number(total['elegiveis'])} elegíveis + {number(total['canceladas'])} canceladas + {number(total['quarentena'])} em quarentena + {number(total['demais_exclusoes'])} demais exclusões.")
        st.write(f"Horários extremos sinalizados: {number(total['horarios_extremos'])}. Eles já integram as elegíveis; não devem ser somados novamente.")
    st.download_button('Baixar indicadores do recorte',monthly.to_csv(index=False).encode('utf-8-sig'),'indicadores_recorte.csv','text/csv',key='baixar_indicadores',width='stretch')

with risk:
    st.subheader('Prioridades de risco')
    st.write('Teste: out–dez/2025. Os filtros recortam a lista global fixa de 19.455 prioridades, sem selecionar novos 10%.')
    rs=risk_summary(filters)
    with st.container(key='risk_metrics'):
        a,b,c,e=st.columns(4)
        a.metric('Recall da lista no recorte',percent(rs['recall']));b.metric('Etapas do teste no recorte',number(rs['n']));c.metric('Etapas prioritárias',number(rs['priorizados']));e.metric('Precisão da lista no recorte',percent(rs['precisao']))
    st.caption(f"{number(rs['acertos'])} atrasos encontrados em {number(rs['priorizados'])} prioridades, de {number(rs['positivos'])} atrasos reais no recorte.")
    if not rs['n']:st.info('Não há etapas do teste final neste recorte. Meses fora de outubro a dezembro permanecem disponíveis em Visão geral.')
    st.subheader('Comparação dos modelos no teste completo')
    comparison=pd.DataFrame([{'Candidato':r['candidato']+(' · escolhido' if r['candidato']==result['escolhido'] else ''),'Recall@10%':percent(r['recall_10']),'Precisão@10%':percent(r['precisao_10']),'Average Precision':round(r['average_precision'],4)} for r in result['teste']])
    with st.container(key='models_desktop'):st.dataframe(comparison,hide_index=True,width='stretch')
    compact=comparison.rename(columns={'Average Precision':'AP'}).copy();compact['AP']=compact['AP'].map(lambda v:f'{v:.4f}'.replace('.',','))
    with st.container(key='models_mobile'):mobile_table(compact)
    st.caption('Tabela fixa do experimento. M2_D6 foi escolhido pela validação, antes de avaliar o teste. Recall@10% no teste: 16,21%; meta exploratória: 20%.')
    priority=priorities(filters)
    with st.expander('Ver etapas priorizadas neste recorte'):
        with st.container(key='priorities_desktop'):st.dataframe(priority[['empresa','numero_voo','origem','destino','partida_prevista','escore','atrasado_30']].head(100),hide_index=True,width='stretch')
        with st.container(key='priorities_mobile'):
            cards=[]
            for row in priority.head(6).itertuples():
                cards.append('<article class="priority-card"><div class="route"><strong>'+escape(row.origem+' → '+row.destino)+'</strong><span>'+f'{row.escore:.4f}'.replace('.',',')+'</span></div><p>'+escape(row.empresa+' '+row.numero_voo)+' · '+pd.Timestamp(row.partida_prevista).strftime('%d/%m/%Y às %H:%M')+'</p><p>Resultado histórico: '+('atraso > 30 min' if row.atrasado_30 else 'até 30 min')+'</p></article>')
            if cards:st.markdown(''.join(cards),unsafe_allow_html=True);st.caption('Até seis etapas na tela. O arquivo CSV contém todas as prioridades do recorte.')
            else:st.info('Sem etapas prioritárias neste recorte.')
    st.download_button('Baixar lista prioritária do recorte',priority.to_csv(index=False).encode('utf-8-sig'),'prioridades_recorte.csv','text/csv',key='baixar_prioridades',width='stretch')

with simulate:
    st.subheader('Explore uma etapa de voo')
    st.write('Escolha a rota e a partida prevista para executar o modelo treinado. A simulação usa somente atributos da programação.')
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
    st.write('Pergunte sobre o histórico, o modelo ou sua última simulação. Os filtros do histórico são usados como contexto.')
    st.caption('Assistente analítico com IA local e respostas vinculadas aos dados do projeto.')
    if 'mensagens' not in st.session_state:st.session_state['mensagens']=[]
    with st.expander('Sugestões de perguntas'):
        suggestions=st.columns(3)
        for col,label in zip(suggestions,['Resumo do recorte','Compare os meses','Qual modelo foi escolhido?']):
            if col.button(label,width='stretch',key='perg_'+label):st.session_state['pergunta_sugerida']=label
    if st.button('Limpar conversa',key='limpar_chat'):st.session_state['mensagens']=[]
    history=st.container(key='chat_history')
    with history:
        for message in st.session_state['mensagens']:
            with st.chat_message('user'):st.write(message['pergunta'])
            with st.chat_message('assistant'):render_reply(message['resposta'])
    prompt=st.chat_input('Ex.: Qual a taxa em agosto na SBGR com a TAM?',key='pergunta_assistente',max_chars=1500)
    question=st.session_state.pop('pergunta_sugerida',None) or prompt
    if question:
        previous=st.session_state['mensagens'][-1]['resposta'] if st.session_state['mensagens'] else None
        reply=answer(question,filters,previous,st.session_state.get('ultima_simulacao'))
        st.session_state['mensagens'].append(dict(pergunta=question,resposta=reply));st.session_state['mensagens']=st.session_state['mensagens'][-20:]
        with history:
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
