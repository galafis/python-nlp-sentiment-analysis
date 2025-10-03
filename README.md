# 🇧🇷 Análise de Sentimentos com NLP Avançado | 🇺🇸 Advanced NLP Sentiment Analysis

<div align="center">

![Hero Image](assets/images/nlp-sentiment-analysis-hero.png)


---

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logoColor=black)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=nltk&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

**Plataforma completa de análise de sentimentos com modelos Transformer e técnicas avançadas de NLP**

[🧠 Modelos](#-modelos-implementados) • [📊 Análises](#-tipos-de-análise) • [⚡ API](#-api-rest) • [🎯 Aplicações](#-aplicações-práticas)

</div>

---

## 🇧🇷 Português

### 🧠 Visão Geral

Plataforma abrangente de **análise de sentimentos e processamento de linguagem natural** desenvolvida em Python:

- 🤖 **Modelos Transformer**: BERT, RoBERTa, DistilBERT, XLNet
- 📊 **Análise Multidimensional**: Sentimento, emoção, aspectos, polaridade
- 🌍 **Multilíngue**: Suporte para português, inglês, espanhol
- 🔄 **Pipeline Completo**: Pré-processamento, análise, visualização
- 🌐 **API REST**: Endpoints para integração em tempo real
- 📈 **Dashboard**: Interface interativa com Streamlit

### 🎯 Objetivos da Plataforma

- **Analisar sentimentos** em textos de múltiplas fontes
- **Extrair insights** de dados textuais não estruturados
- **Monitorar opinião pública** em redes sociais e reviews
- **Automatizar classificação** de feedback de clientes
- **Facilitar tomada de decisão** baseada em análise textual

### 🛠️ Stack Tecnológico

#### NLP e Machine Learning
- **transformers**: Modelos Transformer pré-treinados
- **torch**: PyTorch para deep learning
- **tensorflow**: TensorFlow como alternativa
- **scikit-learn**: Algoritmos de ML clássicos

#### Processamento de Texto
- **spacy**: Processamento avançado de linguagem natural
- **nltk**: Natural Language Toolkit
- **textblob**: Análise de sentimentos simples
- **polyglot**: Processamento multilíngue

#### Análise e Visualização
- **pandas**: Manipulação de dados textuais
- **numpy**: Computação numérica
- **matplotlib**: Visualização de resultados
- **seaborn**: Gráficos estatísticos
- **plotly**: Visualizações interativas
- **wordcloud**: Nuvens de palavras

#### Web e API
- **fastapi**: API REST para análise em tempo real
- **streamlit**: Dashboard interativo
- **uvicorn**: Servidor ASGI
- **requests**: Cliente HTTP

#### Dados e Storage
- **pymongo**: Integração com MongoDB
- **sqlalchemy**: ORM para bancos relacionais
- **redis**: Cache para resultados
- **elasticsearch**: Busca e análise de texto

### 📋 Estrutura da Plataforma

```
python-nlp-sentiment-analysis/
├── 📁 src/                        # Código fonte principal
│   ├── 📁 models/                 # Modelos de análise
│   │   ├── 📁 transformers/       # Modelos Transformer
│   │   │   ├── 📄 bert_sentiment.py # BERT para sentimentos
│   │   │   ├── 📄 roberta_emotion.py # RoBERTa para emoções
│   │   │   ├── 📄 distilbert_aspects.py # DistilBERT aspectos
│   │   │   └── 📄 multilingual_bert.py # BERT multilíngue
│   │   ├── 📁 classical/          # Modelos clássicos
│   │   │   ├── 📄 naive_bayes.py  # Naive Bayes
│   │   │   ├── 📄 svm_classifier.py # SVM
│   │   │   ├── 📄 logistic_regression.py # Regressão logística
│   │   │   └── 📄 random_forest.py # Random Forest
│   │   ├── 📁 ensemble/           # Modelos ensemble
│   │   │   ├── 📄 voting_classifier.py # Voting classifier
│   │   │   ├── 📄 stacking_model.py # Stacking
│   │   │   └── 📄 weighted_ensemble.py # Ensemble ponderado
│   │   └── 📁 custom/             # Modelos customizados
│   │       ├── 📄 lstm_attention.py # LSTM com atenção
│   │       ├── 📄 cnn_text.py     # CNN para texto
│   │       └── 📄 hybrid_model.py # Modelo híbrido
│   ├── 📁 preprocessing/          # Pré-processamento
│   │   ├── 📄 text_cleaner.py     # Limpeza de texto
│   │   ├── 📄 tokenizer.py        # Tokenização
│   │   ├── 📄 feature_extractor.py # Extração de features
│   │   ├── 📄 language_detector.py # Detecção de idioma
│   │   └── 📄 emoji_processor.py  # Processamento de emojis
│   ├── 📁 analysis/               # Módulos de análise
│   │   ├── 📄 sentiment_analyzer.py # Análise de sentimentos
│   │   ├── 📄 emotion_analyzer.py # Análise de emoções
│   │   ├── 📄 aspect_analyzer.py  # Análise de aspectos
│   │   ├── 📄 topic_analyzer.py   # Análise de tópicos
│   │   └── 📄 trend_analyzer.py   # Análise de tendências
│   ├── 📁 data_sources/           # Fontes de dados
│   │   ├── 📄 twitter_collector.py # Coleta do Twitter
│   │   ├── 📄 reddit_collector.py # Coleta do Reddit
│   │   ├── 📄 news_collector.py   # Coleta de notícias
│   │   ├── 📄 review_collector.py # Coleta de reviews
│   │   └── 📄 csv_loader.py       # Carregamento de CSV
│   ├── 📁 visualization/          # Visualização
│   │   ├── 📄 sentiment_plots.py  # Gráficos de sentimento
│   │   ├── 📄 wordcloud_generator.py # Gerador de wordclouds
│   │   ├── 📄 trend_plots.py      # Gráficos de tendência
│   │   ├── 📄 emotion_radar.py    # Radar de emoções
│   │   └── 📄 interactive_plots.py # Plots interativos
│   ├── 📁 api/                    # API REST
│   │   ├── 📄 main.py             # Aplicação FastAPI principal
│   │   ├── 📄 endpoints.py        # Endpoints da API
│   │   ├── 📄 models.py           # Modelos Pydantic
│   │   ├── 📄 dependencies.py     # Dependências
│   │   └── 📄 middleware.py       # Middleware customizado
│   ├── 📁 dashboard/              # Dashboard Streamlit
│   │   ├── 📄 app.py              # Aplicação principal
│   │   ├── 📄 pages/              # Páginas do dashboard
│   │   │   ├── 📄 real_time_analysis.py # Análise em tempo real
│   │   │   ├── 📄 batch_analysis.py # Análise em lote
│   │   │   ├── 📄 model_comparison.py # Comparação de modelos
│   │   │   └── 📄 data_explorer.py # Explorador de dados
│   │   └── 📄 components/         # Componentes reutilizáveis
│   │       ├── 📄 sentiment_meter.py # Medidor de sentimento
│   │       ├── 📄 emotion_chart.py # Gráfico de emoções
│   │       └── 📄 trend_chart.py  # Gráfico de tendências
│   └── 📁 utils/                  # Utilitários
│       ├── 📄 config.py           # Configurações
│       ├── 📄 logger.py           # Sistema de logs
│       ├── 📄 cache.py            # Sistema de cache
│       ├── 📄 metrics.py          # Métricas de avaliação
│       └── 📄 helpers.py          # Funções auxiliares
├── 📁 data/                       # Dados do projeto
│   ├── 📁 raw/                    # Dados brutos
│   │   ├── 📁 twitter/            # Dados do Twitter
│   │   ├── 📁 reviews/            # Reviews de produtos
│   │   ├── 📁 news/               # Artigos de notícias
│   │   └── 📁 surveys/            # Pesquisas e questionários
│   ├── 📁 processed/              # Dados processados
│   │   ├── 📁 cleaned/            # Dados limpos
│   │   ├── 📁 tokenized/          # Dados tokenizados
│   │   └── 📁 features/           # Features extraídas
│   ├── 📁 labeled/                # Dados rotulados
│   │   ├── 📁 sentiment/          # Dados de sentimento
│   │   ├── 📁 emotion/            # Dados de emoção
│   │   └── 📁 aspects/            # Dados de aspectos
│   └── 📁 external/               # Dados externos
│       ├── 📁 lexicons/           # Léxicos de sentimento
│       ├── 📁 embeddings/         # Word embeddings
│       └── 📁 pretrained/         # Modelos pré-treinados
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb # Exploração de dados
│   ├── 📄 02_preprocessing.ipynb  # Pré-processamento
│   ├── 📄 03_classical_models.ipynb # Modelos clássicos
│   ├── 📄 04_transformer_models.ipynb # Modelos Transformer
│   ├── 📄 05_ensemble_methods.ipynb # Métodos ensemble
│   ├── 📄 06_multilingual_analysis.ipynb # Análise multilíngue
│   ├── 📄 07_aspect_based_sentiment.ipynb # Sentimento por aspecto
│   ├── 📄 08_emotion_analysis.ipynb # Análise de emoções
│   ├── 📄 09_trend_analysis.ipynb # Análise de tendências
│   └── 📄 10_model_deployment.ipynb # Deployment de modelos
├── 📁 experiments/                # Experimentos
│   ├── 📁 hyperparameter_tuning/  # Otimização de hiperparâmetros
│   ├── 📁 model_comparison/       # Comparação de modelos
│   ├── 📁 cross_validation/       # Validação cruzada
│   └── 📁 ablation_studies/       # Estudos de ablação
├── 📁 models/                     # Modelos treinados
│   ├── 📁 sentiment/              # Modelos de sentimento
│   ├── 📁 emotion/                # Modelos de emoção
│   ├── 📁 aspects/                # Modelos de aspectos
│   └── 📁 multilingual/           # Modelos multilíngues
├── 📁 tests/                      # Testes automatizados
│   ├── 📄 test_preprocessing.py   # Testes pré-processamento
│   ├── 📄 test_models.py          # Testes de modelos
│   ├── 📄 test_api.py             # Testes da API
│   └── 📄 test_analysis.py        # Testes de análise
├── 📁 docker/                     # Containers Docker
│   ├── 📄 Dockerfile.api          # Container da API
│   ├── 📄 Dockerfile.dashboard    # Container do dashboard
│   └── 📄 docker-compose.yml      # Orquestração
├── 📁 configs/                    # Configurações
│   ├── 📄 model_configs.yaml      # Configurações de modelos
│   ├── 📄 api_config.yaml         # Configuração da API
│   └── 📄 data_sources.yaml       # Configuração de fontes
├── 📄 requirements.txt            # Dependências Python
├── 📄 requirements-dev.txt        # Dependências desenvolvimento
├── 📄 setup.py                    # Setup do pacote
├── 📄 README.md                   # Este arquivo
├── 📄 LICENSE                     # Licença MIT
└── 📄 .gitignore                 # Arquivos ignorados
```

### 🧠 Modelos Implementados

#### 1. 🤖 Modelos Transformer

**BERT para Análise de Sentimentos**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BERTSentimentAnalyzer:
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sentiment(self, text, return_probabilities=True):
        """Predizer sentimento de um texto"""
        # Tokenização
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=-1)
                return {
                    'sentiment': self._get_sentiment_label(logits),
                    'confidence': float(torch.max(probabilities)),
                    'probabilities': {
                        'negative': float(probabilities[0][0]),
                        'neutral': float(probabilities[0][1]),
                        'positive': float(probabilities[0][2])
                    }
                }
            else:
                return self._get_sentiment_label(logits)
    
    def batch_predict(self, texts, batch_size=32):
        """Predição em lote para múltiplos textos"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenização do lote
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                for j, text in enumerate(batch_texts):
                    results.append({
                        'text': text,
                        'sentiment': self._get_sentiment_label(logits[j:j+1]),
                        'confidence': float(torch.max(probabilities[j])),
                        'probabilities': {
                            'negative': float(probabilities[j][0]),
                            'neutral': float(probabilities[j][1]),
                            'positive': float(probabilities[j][2])
                        }
                    })
        
        return results
    
    def _get_sentiment_label(self, logits):
        """Converter logits para rótulo de sentimento"""
        predicted_class = torch.argmax(logits, dim=-1).item()
        labels = ['negative', 'neutral', 'positive']
        return labels[predicted_class]
    
    def fine_tune(self, train_dataset, val_dataset, epochs=3, learning_rate=2e-5):
        """Fine-tuning do modelo com dados customizados"""
        from torch.utils.data import DataLoader
        from transformers import AdamW, get_linear_schedule_with_warmup
        
        # Configurar otimizador
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Configurar scheduler
        total_steps = len(train_dataset) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            # Validação
            val_accuracy = self._evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
```

**RoBERTa para Análise de Emoções**
```python
class RoBERTaEmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Mapeamento de emoções
        self.emotion_labels = [
            'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'
        ]
    
    def analyze_emotions(self, text):
        """Analisar emoções em um texto"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Criar dicionário de emoções com probabilidades
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = float(probabilities[0][i])
            
            # Emoção dominante
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion],
                'all_emotions': emotion_scores,
                'emotion_intensity': self._calculate_intensity(emotion_scores)
            }
    
    def _calculate_intensity(self, emotion_scores):
        """Calcular intensidade emocional geral"""
        # Excluir emoções neutras e calcular intensidade
        intense_emotions = ['anger', 'fear', 'sadness', 'joy', 'love', 'surprise']
        total_intensity = sum(emotion_scores[emotion] for emotion in intense_emotions)
        return total_intensity
    
    def emotion_timeline(self, texts, timestamps=None):
        """Analisar evolução emocional ao longo do tempo"""
        results = []
        
        for i, text in enumerate(texts):
            emotion_analysis = self.analyze_emotions(text)
            
            result = {
                'index': i,
                'text': text,
                'timestamp': timestamps[i] if timestamps else i,
                **emotion_analysis
            }
            results.append(result)
        
        return results
```

#### 2. 📊 Análise de Aspectos (ABSA)

**Aspect-Based Sentiment Analysis**
```python
import spacy
from collections import defaultdict

class AspectBasedSentimentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_sm")
        self.sentiment_analyzer = BERTSentimentAnalyzer()
        
        # Aspectos pré-definidos para diferentes domínios
        self.aspect_keywords = {
            'restaurant': {
                'food': ['comida', 'prato', 'sabor', 'tempero', 'ingrediente'],
                'service': ['atendimento', 'garçom', 'serviço', 'staff'],
                'ambiance': ['ambiente', 'decoração', 'música', 'iluminação'],
                'price': ['preço', 'valor', 'custo', 'caro', 'barato']
            },
            'hotel': {
                'room': ['quarto', 'cama', 'banheiro', 'limpeza'],
                'service': ['atendimento', 'recepção', 'staff'],
                'location': ['localização', 'local', 'acesso', 'transporte'],
                'amenities': ['wifi', 'piscina', 'academia', 'café']
            },
            'product': {
                'quality': ['qualidade', 'material', 'durabilidade'],
                'design': ['design', 'aparência', 'cor', 'estilo'],
                'functionality': ['funcionalidade', 'performance', 'uso'],
                'price': ['preço', 'valor', 'custo-benefício']
            }
        }
    
    def extract_aspects_and_sentiments(self, text, domain='general'):
        """Extrair aspectos e seus sentimentos"""
        doc = self.nlp(text)
        
        # Extrair aspectos mencionados
        aspects_found = self._extract_aspects(text, domain)
        
        # Analisar sentimento para cada aspecto
        aspect_sentiments = {}
        
        for aspect, mentions in aspects_found.items():
            if mentions:
                # Extrair sentenças que mencionam o aspecto
                aspect_sentences = self._extract_aspect_sentences(text, mentions)
                
                # Analisar sentimento das sentenças
                sentiments = []
                for sentence in aspect_sentences:
                    sentiment = self.sentiment_analyzer.predict_sentiment(sentence)
                    sentiments.append(sentiment)
                
                # Agregar sentimentos
                aspect_sentiments[aspect] = self._aggregate_sentiments(sentiments)
        
        return {
            'aspects_found': aspects_found,
            'aspect_sentiments': aspect_sentiments,
            'overall_sentiment': self.sentiment_analyzer.predict_sentiment(text)
        }
    
    def _extract_aspects(self, text, domain):
        """Extrair aspectos mencionados no texto"""
        text_lower = text.lower()
        aspects_found = defaultdict(list)
        
        if domain in self.aspect_keywords:
            for aspect, keywords in self.aspect_keywords[domain].items():
                for keyword in keywords:
                    if keyword in text_lower:
                        aspects_found[aspect].append(keyword)
        
        return dict(aspects_found)
    
    def _extract_aspect_sentences(self, text, aspect_mentions):
        """Extrair sentenças que mencionam aspectos específicos"""
        doc = self.nlp(text)
        aspect_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for mention in aspect_mentions:
                if mention in sent_text:
                    aspect_sentences.append(sent.text)
                    break
        
        return aspect_sentences
    
    def _aggregate_sentiments(self, sentiments):
        """Agregar múltiplos sentimentos para um aspecto"""
        if not sentiments:
            return None
        
        # Calcular médias das probabilidades
        avg_probs = {
            'negative': sum(s['probabilities']['negative'] for s in sentiments) / len(sentiments),
            'neutral': sum(s['probabilities']['neutral'] for s in sentiments) / len(sentiments),
            'positive': sum(s['probabilities']['positive'] for s in sentiments) / len(sentiments)
        }
        
        # Determinar sentimento dominante
        dominant_sentiment = max(avg_probs, key=avg_probs.get)
        
        return {
            'sentiment': dominant_sentiment,
            'confidence': avg_probs[dominant_sentiment],
            'probabilities': avg_probs,
            'num_mentions': len(sentiments)
        }
```

#### 3. 🌍 Análise Multilíngue

**Detector de Idioma e Análise Multilíngue**
```python
from langdetect import detect, detect_langs
import polyglot
from polyglot.text import Text

class MultilingualSentimentAnalyzer:
    def __init__(self):
        # Modelos específicos por idioma
        self.models = {
            'pt': BERTSentimentAnalyzer("neuralmind/bert-base-portuguese-cased"),
            'en': BERTSentimentAnalyzer("cardiffnlp/twitter-roberta-base-sentiment-latest"),
            'es': BERTSentimentAnalyzer("pysentimiento/robertuito-sentiment-analysis"),
            'fr': BERTSentimentAnalyzer("tblard/tf-allocine"),
        }
        
        # Modelo multilíngue como fallback
        self.multilingual_model = BERTSentimentAnalyzer("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    def detect_language(self, text):
        """Detectar idioma do texto"""
        try:
            # Detecção principal
            language = detect(text)
            
            # Detecção com probabilidades
            lang_probs = detect_langs(text)
            
            return {
                'language': language,
                'confidence': lang_probs[0].prob,
                'all_languages': [(lang.lang, lang.prob) for lang in lang_probs]
            }
        except:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'all_languages': []
            }
    
    def analyze_multilingual_sentiment(self, text):
        """Analisar sentimento considerando o idioma"""
        # Detectar idioma
        lang_info = self.detect_language(text)
        detected_lang = lang_info['language']
        
        # Escolher modelo apropriado
        if detected_lang in self.models and lang_info['confidence'] > 0.8:
            model = self.models[detected_lang]
            model_used = f"language_specific_{detected_lang}"
        else:
            model = self.multilingual_model
            model_used = "multilingual_fallback"
        
        # Analisar sentimento
        sentiment_result = model.predict_sentiment(text)
        
        # Adicionar informações de idioma
        sentiment_result.update({
            'detected_language': detected_lang,
            'language_confidence': lang_info['confidence'],
            'model_used': model_used
        })
        
        return sentiment_result
    
    def cross_lingual_analysis(self, texts_dict):
        """Análise comparativa entre idiomas"""
        results = {}
        
        for lang, texts in texts_dict.items():
            lang_results = []
            
            for text in texts:
                result = self.analyze_multilingual_sentiment(text)
                lang_results.append(result)
            
            # Agregar resultados por idioma
            results[lang] = {
                'individual_results': lang_results,
                'aggregated': self._aggregate_language_results(lang_results)
            }
        
        return results
    
    def _aggregate_language_results(self, results):
        """Agregar resultados de um idioma"""
        if not results:
            return None
        
        # Calcular distribuição de sentimentos
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_confidence = 0
        
        for result in results:
            sentiment_counts[result['sentiment']] += 1
            total_confidence += result['confidence']
        
        total_texts = len(results)
        
        return {
            'total_texts': total_texts,
            'sentiment_distribution': {
                k: v/total_texts for k, v in sentiment_counts.items()
            },
            'average_confidence': total_confidence / total_texts,
            'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
        }
```

### 🌐 API REST

**FastAPI para Análise em Tempo Real**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(
    title="NLP Sentiment Analysis API",
    description="API completa para análise de sentimentos e processamento de linguagem natural",
    version="1.0.0"
)

# Modelos Pydantic
class TextInput(BaseModel):
    text: str
    language: Optional[str] = None
    domain: Optional[str] = "general"

class BatchTextInput(BaseModel):
    texts: List[str]
    language: Optional[str] = None
    domain: Optional[str] = "general"

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    processing_time: float

# Inicializar analisadores
sentiment_analyzer = BERTSentimentAnalyzer()
emotion_analyzer = RoBERTaEmotionAnalyzer()
aspect_analyzer = AspectBasedSentimentAnalyzer()
multilingual_analyzer = MultilingualSentimentAnalyzer()

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """Analisar sentimento de um texto"""
    import time
    start_time = time.time()
    
    try:
        if input_data.language:
            # Usar modelo específico do idioma se especificado
            result = multilingual_analyzer.analyze_multilingual_sentiment(input_data.text)
        else:
            # Usar modelo padrão
            result = sentiment_analyzer.predict_sentiment(input_data.text)
        
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/emotions")
async def analyze_emotions(input_data: TextInput):
    """Analisar emoções de um texto"""
    try:
        result = emotion_analyzer.analyze_emotions(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/aspects")
async def analyze_aspects(input_data: TextInput):
    """Análise de sentimentos baseada em aspectos"""
    try:
        result = aspect_analyzer.extract_aspects_and_sentiments(
            input_data.text, 
            input_data.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analyze(input_data: BatchTextInput):
    """Análise em lote de múltiplos textos"""
    try:
        results = sentiment_analyzer.batch_predict(input_data.texts)
        return {"results": results, "total_processed": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/comprehensive")
async def comprehensive_analysis(input_data: TextInput):
    """Análise completa: sentimento, emoções e aspectos"""
    try:
        # Executar análises em paralelo
        sentiment_task = asyncio.create_task(
            asyncio.to_thread(sentiment_analyzer.predict_sentiment, input_data.text)
        )
        emotion_task = asyncio.create_task(
            asyncio.to_thread(emotion_analyzer.analyze_emotions, input_data.text)
        )
        aspect_task = asyncio.create_task(
            asyncio.to_thread(
                aspect_analyzer.extract_aspects_and_sentiments, 
                input_data.text, 
                input_data.domain
            )
        )
        
        # Aguardar resultados
        sentiment_result = await sentiment_task
        emotion_result = await emotion_task
        aspect_result = await aspect_task
        
        return {
            "sentiment_analysis": sentiment_result,
            "emotion_analysis": emotion_result,
            "aspect_analysis": aspect_result,
            "text": input_data.text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {"status": "healthy", "models_loaded": True}

# Middleware para logging
@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Path: {request.url.path}, Method: {request.method}, Time: {process_time:.4f}s")
    return response
```

### 📊 Dashboard Streamlit

**Interface Interativa**
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd

st.set_page_config(
    page_title="NLP Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide"
)

class SentimentDashboard:
    def __init__(self):
        self.sentiment_analyzer = BERTSentimentAnalyzer()
        self.emotion_analyzer = RoBERTaEmotionAnalyzer()
        self.aspect_analyzer = AspectBasedSentimentAnalyzer()
    
    def main(self):
        st.title("🧠 NLP Sentiment Analysis Dashboard")
        st.sidebar.title("Navegação")
        
        page = st.sidebar.selectbox(
            "Escolha uma análise",
            ["Análise Individual", "Análise em Lote", "Análise de Aspectos", 
             "Análise de Emoções", "Comparação de Modelos", "Análise Temporal"]
        )
        
        if page == "Análise Individual":
            self.individual_analysis()
        elif page == "Análise em Lote":
            self.batch_analysis()
        elif page == "Análise de Aspectos":
            self.aspect_analysis()
        elif page == "Análise de Emoções":
            self.emotion_analysis()
        elif page == "Comparação de Modelos":
            self.model_comparison()
        elif page == "Análise Temporal":
            self.temporal_analysis()
    
    def individual_analysis(self):
        st.header("📝 Análise Individual de Texto")
        
        # Input de texto
        text_input = st.text_area(
            "Digite o texto para análise:",
            height=150,
            placeholder="Ex: Adorei o produto! A qualidade é excelente e o atendimento foi perfeito."
        )
        
        if st.button("Analisar Sentimento") and text_input:
            with st.spinner("Analisando..."):
                # Análise de sentimento
                sentiment_result = self.sentiment_analyzer.predict_sentiment(text_input)
                
                # Exibir resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentimento",
                        sentiment_result['sentiment'].title(),
                        f"{sentiment_result['confidence']:.2%}"
                    )
                
                with col2:
                    # Gráfico de barras das probabilidades
                    probs_df = pd.DataFrame([sentiment_result['probabilities']]).T
                    probs_df.columns = ['Probabilidade']
                    fig = px.bar(probs_df, y=probs_df.index, x='Probabilidade', 
                               orientation='h', title="Distribuição de Probabilidades")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    # Medidor de sentimento
                    self.sentiment_gauge(sentiment_result['probabilities'])
    
    def sentiment_gauge(self, probabilities):
        """Criar medidor de sentimento"""
        # Calcular score (-1 a 1)
        score = probabilities['positive'] - probabilities['negative']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score de Sentimento"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightyellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def batch_analysis(self):
        st.header("📊 Análise em Lote")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Faça upload de um arquivo CSV com textos",
            type=['csv']
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview dos dados:")
            st.dataframe(df.head())
            
            text_column = st.selectbox(
                "Selecione a coluna com os textos:",
                df.columns
            )
            
            if st.button("Analisar Todos os Textos"):
                with st.spinner("Processando..."):
                    # Análise em lote
                    texts = df[text_column].tolist()
                    results = self.sentiment_analyzer.batch_predict(texts)
                    
                    # Criar DataFrame com resultados
                    results_df = pd.DataFrame(results)
                    
                    # Estatísticas gerais
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total de Textos", len(results))
                    
                    with col2:
                        positive_pct = len([r for r in results if r['sentiment'] == 'positive']) / len(results)
                        st.metric("% Positivos", f"{positive_pct:.1%}")
                    
                    with col3:
                        avg_confidence = sum(r['confidence'] for r in results) / len(results)
                        st.metric("Confiança Média", f"{avg_confidence:.2%}")
                    
                    # Gráficos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribuição de sentimentos
                        sentiment_counts = results_df['sentiment'].value_counts()
                        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                                   title="Distribuição de Sentimentos")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Histograma de confiança
                        fig = px.histogram(results_df, x='confidence', nbins=20,
                                         title="Distribuição de Confiança")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela de resultados
                    st.subheader("Resultados Detalhados")
                    st.dataframe(results_df)
                    
                    # Download dos resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download dos Resultados",
                        csv,
                        "sentiment_analysis_results.csv",
                        "text/csv"
                    )

# Executar dashboard
if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.main()
```

### 🎯 Aplicações Práticas

#### 1. 📱 Monitoramento de Redes Sociais

**Análise de Sentimento no Twitter**
```python
class SocialMediaMonitor:
    def __init__(self):
        self.sentiment_analyzer = MultilingualSentimentAnalyzer()
        self.emotion_analyzer = RoBERTaEmotionAnalyzer()
    
    def monitor_brand_sentiment(self, brand_name, num_tweets=1000):
        """Monitorar sentimento sobre uma marca"""
        # Coletar tweets (simulado)
        tweets = self._collect_tweets(brand_name, num_tweets)
        
        # Analisar sentimentos
        results = []
        for tweet in tweets:
            sentiment = self.sentiment_analyzer.analyze_multilingual_sentiment(tweet['text'])
            emotion = self.emotion_analyzer.analyze_emotions(tweet['text'])
            
            results.append({
                'tweet_id': tweet['id'],
                'text': tweet['text'],
                'timestamp': tweet['timestamp'],
                'user': tweet['user'],
                'sentiment': sentiment['sentiment'],
                'sentiment_confidence': sentiment['confidence'],
                'dominant_emotion': emotion['dominant_emotion'],
                'emotion_intensity': emotion['emotion_intensity']
            })
        
        # Análise agregada
        analysis = self._aggregate_social_analysis(results)
        
        return {
            'brand': brand_name,
            'total_mentions': len(results),
            'individual_results': results,
            'aggregated_analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
    
    def _aggregate_social_analysis(self, results):
        """Agregar análise de redes sociais"""
        if not results:
            return None
        
        # Distribuição de sentimentos
        sentiment_dist = {}
        emotion_dist = {}
        
        for result in results:
            sentiment = result['sentiment']
            emotion = result['dominant_emotion']
            
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # Calcular métricas
        total = len(results)
        positive_ratio = sentiment_dist.get('positive', 0) / total
        negative_ratio = sentiment_dist.get('negative', 0) / total
        
        # Score de sentimento da marca
        brand_sentiment_score = positive_ratio - negative_ratio
        
        return {
            'sentiment_distribution': sentiment_dist,
            'emotion_distribution': emotion_dist,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'brand_sentiment_score': brand_sentiment_score,
            'avg_sentiment_confidence': sum(r['sentiment_confidence'] for r in results) / total,
            'avg_emotion_intensity': sum(r['emotion_intensity'] for r in results) / total
        }
```

#### 2. 🛍️ Análise de Reviews de Produtos

**Sistema de Análise de Reviews**
```python
class ProductReviewAnalyzer:
    def __init__(self):
        self.aspect_analyzer = AspectBasedSentimentAnalyzer()
        self.sentiment_analyzer = BERTSentimentAnalyzer()
    
    def analyze_product_reviews(self, product_id, reviews):
        """Analisar reviews de um produto"""
        analysis_results = []
        
        for review in reviews:
            # Análise completa do review
            result = {
                'review_id': review['id'],
                'rating': review['rating'],
                'text': review['text'],
                'date': review['date'],
                'overall_sentiment': self.sentiment_analyzer.predict_sentiment(review['text']),
                'aspect_analysis': self.aspect_analyzer.extract_aspects_and_sentiments(
                    review['text'], 'product'
                )
            }
            analysis_results.append(result)
        
        # Análise agregada
        product_insights = self._generate_product_insights(analysis_results)
        
        return {
            'product_id': product_id,
            'total_reviews': len(reviews),
            'individual_analyses': analysis_results,
            'product_insights': product_insights,
            'improvement_suggestions': self._suggest_improvements(product_insights)
        }
    
    def _generate_product_insights(self, analyses):
        """Gerar insights do produto"""
        if not analyses:
            return None
        
        # Análise por aspecto
        aspect_insights = {}
        overall_sentiments = []
        
        for analysis in analyses:
            overall_sentiments.append(analysis['overall_sentiment']['sentiment'])
            
            for aspect, sentiment_data in analysis['aspect_analysis']['aspect_sentiments'].items():
                if aspect not in aspect_insights:
                    aspect_insights[aspect] = []
                aspect_insights[aspect].append(sentiment_data['sentiment'])
        
        # Calcular scores por aspecto
        aspect_scores = {}
        for aspect, sentiments in aspect_insights.items():
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            total_count = len(sentiments)
            
            aspect_scores[aspect] = {
                'positive_ratio': positive_count / total_count,
                'negative_ratio': negative_count / total_count,
                'total_mentions': total_count,
                'score': (positive_count - negative_count) / total_count
            }
        
        # Score geral do produto
        positive_overall = overall_sentiments.count('positive')
        negative_overall = overall_sentiments.count('negative')
        total_overall = len(overall_sentiments)
        
        overall_score = (positive_overall - negative_overall) / total_overall
        
        return {
            'overall_score': overall_score,
            'overall_sentiment_distribution': {
                'positive': positive_overall / total_overall,
                'negative': negative_overall / total_overall,
                'neutral': overall_sentiments.count('neutral') / total_overall
            },
            'aspect_scores': aspect_scores,
            'strongest_aspects': self._get_top_aspects(aspect_scores, 'positive'),
            'weakest_aspects': self._get_top_aspects(aspect_scores, 'negative')
        }
    
    def _suggest_improvements(self, insights):
        """Sugerir melhorias baseadas na análise"""
        suggestions = []
        
        # Identificar aspectos problemáticos
        for aspect, scores in insights['aspect_scores'].items():
            if scores['score'] < -0.3 and scores['total_mentions'] >= 5:
                suggestions.append({
                    'priority': 'high',
                    'aspect': aspect,
                    'issue': f"Aspecto {aspect} tem sentimento predominantemente negativo",
                    'recommendation': f"Focar em melhorias no aspecto {aspect}"
                })
        
        # Sugestões baseadas no score geral
        if insights['overall_score'] < 0:
            suggestions.append({
                'priority': 'critical',
                'aspect': 'overall',
                'issue': "Sentimento geral do produto é negativo",
                'recommendation': "Revisar estratégia geral do produto"
            })
        
        return suggestions
```

### 🎯 Competências Demonstradas

#### NLP e Deep Learning
- ✅ **Modelos Transformer**: BERT, RoBERTa, DistilBERT, XLNet
- ✅ **Transfer Learning**: Fine-tuning para domínios específicos
- ✅ **Multilingual NLP**: Processamento em múltiplos idiomas
- ✅ **Aspect-Based Sentiment**: Análise granular por aspectos

#### Análise de Texto
- ✅ **Sentiment Analysis**: Classificação de polaridade
- ✅ **Emotion Analysis**: Detecção de emoções específicas
- ✅ **Topic Modeling**: Extração de tópicos
- ✅ **Text Classification**: Classificação automática

#### Engenharia de Software
- ✅ **API Development**: FastAPI para produção
- ✅ **Dashboard Development**: Streamlit para visualização
- ✅ **Batch Processing**: Processamento em lote eficiente
- ✅ **Model Deployment**: Deploy de modelos em produção

---

## 🇺🇸 English

### 🧠 Overview

Comprehensive **sentiment analysis and natural language processing** platform developed in Python:

- 🤖 **Transformer Models**: BERT, RoBERTa, DistilBERT, XLNet
- 📊 **Multidimensional Analysis**: Sentiment, emotion, aspects, polarity
- 🌍 **Multilingual**: Support for Portuguese, English, Spanish
- 🔄 **Complete Pipeline**: Preprocessing, analysis, visualization
- 🌐 **REST API**: Endpoints for real-time integration
- 📈 **Dashboard**: Interactive interface with Streamlit

### 🎯 Platform Objectives

- **Analyze sentiments** in texts from multiple sources
- **Extract insights** from unstructured textual data
- **Monitor public opinion** on social media and reviews
- **Automate classification** of customer feedback
- **Facilitate decision-making** based on textual analysis

### 🧠 Implemented Models

#### 1. 🤖 Transformer Models
- BERT for sentiment analysis
- RoBERTa for emotion analysis
- DistilBERT for aspect analysis
- Multilingual BERT for cross-language analysis

#### 2. 📊 Aspect Analysis (ABSA)
- Aspect extraction from text
- Sentiment analysis per aspect
- Domain-specific aspect detection
- Aggregated aspect insights

#### 3. 🌍 Multilingual Analysis
- Language detection
- Language-specific models
- Cross-lingual comparison
- Multilingual fallback models

### 🎯 Skills Demonstrated

#### NLP and Deep Learning
- ✅ **Transformer Models**: BERT, RoBERTa, DistilBERT, XLNet
- ✅ **Transfer Learning**: Fine-tuning for specific domains
- ✅ **Multilingual NLP**: Multi-language processing
- ✅ **Aspect-Based Sentiment**: Granular aspect analysis

#### Text Analysis
- ✅ **Sentiment Analysis**: Polarity classification
- ✅ **Emotion Analysis**: Specific emotion detection
- ✅ **Topic Modeling**: Topic extraction
- ✅ **Text Classification**: Automatic classification

#### Software Engineering
- ✅ **API Development**: FastAPI for production
- ✅ **Dashboard Development**: Streamlit for visualization
- ✅ **Batch Processing**: Efficient batch processing
- ✅ **Model Deployment**: Production model deployment

---

## 📄 Licença | License

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes | see [LICENSE](LICENSE) file for details

## 📞 Contato | Contact

**GitHub**: [@galafis](https://github.com/galafis)  
**LinkedIn**: [Gabriel Demetrios Lafis](https://linkedin.com/in/galafis)  
**Email**: gabriel.lafis@example.com

---

<div align="center">

**Desenvolvido com ❤️ para Processamento de Linguagem Natural | Developed with ❤️ for Natural Language Processing**

[![GitHub](https://img.shields.io/badge/GitHub-galafis-blue?style=flat-square&logo=github)](https://github.com/galafis)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

