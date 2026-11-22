import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def avaliar_decisao(df, decisao='decisao_020', corte=None, coluna_real=None):
  """Monta matriz de confusão e métricas (Precisão e Sensibilidade/Recall)

  para uma regra de decisão ('decisao_020', 'decisao_010', etc.) ou corte
  numérico.
  """
  # 1. Identifica a coluna real ('y_real' ou 'y')
  if coluna_real is None:
    if 'y_real' in df.columns:
      coluna_real = 'y_real'
    elif 'y' in df.columns:
      coluna_real = 'y'
    else:
      raise KeyError(
          "Não encontrei coluna real ('y_real' ou 'y') em bank_metricas."
      )

  y_true = pd.to_numeric(df[coluna_real], errors='coerce').astype(int)

  # 2. Define a coluna/regra de decisão
  if isinstance(decisao, (int, float)):
    # Se passaram um número direto (ex.: 0.20), cria a coluna correspondente
    corte_num = float(decisao)
    nome_col = f'decisao_{int(round(corte_num * 100)):03d}'
    base_prob = (
        df['prob_y'] if 'prob_y' in df.columns else df['y_chapeu']
    )
    df[nome_col] = np.where(base_prob >= corte_num, 1, 0)
    decisao = nome_col
  else:
    if decisao not in df.columns:
      # Tenta inferir o corte pelo sufixo, ex.: 'decisao_020' -> 0.20
      if corte is None:
        try:
          corte = int(decisao.split('_')[-1]) / 100.0
        except (ValueError, IndexError) as exc:
          raise KeyError(
              f"A coluna '{decisao}' não existe e não foi possível inferir o"
              ' corte.'
          ) from exc
      base_prob = (
          df['prob_y'] if 'prob_y' in df.columns else df['y_chapeu']
      )
      df[decisao] = np.where(base_prob >= corte, 1, 0)

  y_pred = pd.to_numeric(df[decisao], errors='coerce').astype(int)

  # 3. Matriz de confusão (linhas = real, colunas = previsto: 0 e 1)
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  matriz_df = pd.DataFrame(
      cm,
      index=['Real = 0', 'Real = 1'],
      columns=['Previsto = 0', 'Previsto = 1'],
  )

  titulo_corte = f' (corte >= {corte:.2f})' if corte is not None else ''
  print(f'=== Matriz de Confusão — {decisao}{titulo_corte} ===')
  display(matriz_df)

  # 4. Extrai VN, FP, FN, VP e métricas
  vn, fp, fn, vp = cm.ravel()
  precisao = precision_score(y_true, y_pred, zero_division=0)
  recall = recall_score(y_true, y_pred, zero_division=0)

  resumo_metricas = pd.DataFrame({
      'Regra': [decisao],
      'VN (0 real / 0 prev)': [vn],
      'FP (0 real / 1 prev)': [fp],
      'FN (1 real / 0 prev)': [fn],
      'VP (1 real / 1 prev)': [vp],
      'Precisão': [round(precisao, 4)],
      'Sensibilidade / Recall': [round(recall, 4)],
  })

  display(resumo_metricas)
  return matriz_df, resumo_metricas


# Exemplos de uso:
# matriz_020, resumo_020 = avaliar_decisao(bank_metricas, decisao='decisao_020')
# matriz_010, resumo_010 = avaliar_decisao(bank_metricas, decisao='decisao_010')
# matriz_015, resumo_015 = avaliar_decisao(bank_metricas, decisao=0.15)
