# Pontualidade em foco

Aplicação Streamlit do Projeto Aplicado: indicadores do VRA/ANAC de 2025,
ranking histórico de risco, inferência no modelo treinado e assistente
analítico com IA local. Não exige chave de API ou inferência externa.

## Publicar no Streamlit Community Cloud

1. Envie o conteúdo desta pasta a um repositório GitHub de sua conta.
   Preserve `artefatos_app/`, `.streamlit/` e `.gitattributes`.
2. Entre em https://share.streamlit.io e conecte o repositório.
3. Clique em **Create app** e escolha **Yup, I have an app**.
4. Selecione o repositório, a branch **main** e o arquivo
   **app_streamlit.py** na raiz.
5. Em **Advanced settings**, selecione **Python 3.11**.
6. Não é necessário preencher **Secrets**. Clique em **Deploy**.

A aplicação recebe uma URL terminada em `streamlit.app`. O subdomínio pode
ser escolhido na publicação, conforme disponibilidade. Revise a configuração
de acesso do aplicativo para definir quem poderá visitar a URL.

O Python 3.11 e as versões de `requirements.txt` correspondem ao ambiente
dos artefatos treinados. Preserve essas versões na primeira publicação.
Os modelos já estão treinados; a inicialização não processa os CSVs anuais
nem executa novamente o experimento preditivo.

## Executar localmente

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app_streamlit.py
```

No Linux, use `.venv/bin/python` no lugar de `.venv/Scripts/python.exe`.

## Testes

```text
python -m unittest discover -p test_produto_final.py -v
```

Os 16 testes verificam consultas, filtros, reconciliação, inferência,
continuação de conversa e interface via Streamlit AppTest.

## Arquivos

- `app_streamlit.py`: interface com cinco abas.
- `produto_voos.py`: dados, consultas e inferência.
- `assistente_voos.py`: interpretação de intenções e respostas fundamentadas.
- `artefatos_app/`: dados de consumo, modelos, encoder, corpus e manifesto.
- `requirements.txt`: dependências com versões fixas.

O diretório `artefatos_app` deve permanecer byte a byte igual ao manifesto.
O `.gitattributes` impede conversão de quebras de linha nesses arquivos:
alterar seus bytes faria a verificação de integridade rejeitar os dados.
Os maiores arquivos têm menos de 3 MB e não exigem Git LFS.

## Dados e alcance

Fonte: Voo Regular Ativo, ANAC/SIROS, janeiro a dezembro/2025:
https://siros.anac.gov.br/siros/registros/diversos/vra/2025/

A base descritiva contém 769.740 etapas elegíveis e 56.600 atrasos > 30 min.
O modelo M2_D6 foi escolhido na validação; recall@10% no teste: 16,21%,
abaixo da meta exploratória de 20%. Os escores não são probabilidades
calibradas e não demonstram previsão operacional em tempo real.

O assistente usa TF-IDF e regressão logística para 12 intenções. Os números
vêm de consultas locais; não é um LLM generativo. A classificação acertou
20 de 24 perguntas sintéticas reservadas (83,33%). A avaliação realizada é
técnica e não incluiu estudo com participantes ou medição de benefício humano.

Projeto acadêmico de Marco Antonio Luiz Giampietro Junior — XP Educação.

## Documentação de publicação

- [Deploy no Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Dependências do aplicativo](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
