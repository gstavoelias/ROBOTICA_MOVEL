# Treinamento e Uso de Modelos com YOLO (Ultralytics)

## 📦 Pré-requisitos — Estrutura de Dados

Para treinar e validar modelos usando o YOLO da Ultralytics, é necessário organizar os dados no seguinte formato:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
├── validation/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
├── data.yaml
```

**Explicação:**
- `images/` contém as imagens de entrada.
- `labels/` contém os arquivos `.txt` com as anotações no formato YOLO (classe x_center y_center width height).
- `data.yaml` descreve o caminho dos dados e as classes.

Exemplo de um `data.yaml`:

```yaml
train: /caminho/para/dataset/train/images
val: /caminho/para/dataset/validation/images

nc: 2  # número de classes
names: ['classe1', 'classe2']
```

---

## 🚀 Uso do Ultralytics YOLO

Primeiro, instale a biblioteca ultralytics:

```bash
pip install ultralytics
```

Importe e carregue o modelo:

```python
from ultralytics import YOLO

# Carregando um modelo pré-treinado ou personalizado
model = YOLO('yolov8n.pt')  # ou 'caminho/para/seu/modelo.pt'
```

---

## 🧀 Treinar (train)

Para treinar o modelo em seus próprios dados:

```python
model.train(
    data='caminho/para/data.yaml',
    epochs=100,
    imgsz=640
)
```

**Parâmetros comuns:**
- `data`: caminho para o arquivo `data.yaml`.
- `epochs`: número de épocas de treinamento.
- `imgsz`: tamanho das imagens.

---

## 🔎 Avaliar (val)

Para validar o modelo usando o conjunto de validação:

```python
model.val(
    data='caminho/para/data.yaml'
)
```

Isso gera métricas como precisão (precision), recall e mAP.

---

## 🎯 Fazer predições (predict)

Para usar o modelo treinado em novas imagens:

```python
model.predict(
    source='caminho/para/imagens',  # pode ser uma pasta, imagem única, ou URL
    save=True
)
```

**Parâmetros comuns:**
- `source`: caminho para as imagens a serem analisadas.
- `conf`: confiança mínima para considerar uma detecção (ex.: `conf=0.25`).
- `save`: se deve salvar as imagens com as detecções.

---

# ✅ Resumo

| Ação | Comando |
|:---|:---|
| Treinar | `model.train(...)` |
| Validar | `model.val(...)` |
| Predizer | `model.predict(...)` |

---

# Observação
- A Ultralytics recomenda sempre **usar imagens e labels limpas e corretamente anotadas** para melhor desempenho.
- Use **augmentações** e **validações cruzadas** para melhores resultados em datasets pequenos.
