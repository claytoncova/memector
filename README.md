# 🎥 Memector - Detector de Vídeos Gerados por IA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![AI Assisted](https://img.shields.io/badge/AI%20Assisted-🤖-purple)

</div>

## 📝 Sobre o Projeto

O Memector é uma ferramenta de código aberto desenvolvida para identificar possíveis indícios de vídeos gerados por Inteligência Artificial. Utilizando técnicas avançadas de processamento de imagem e análise de vídeo, o Memector analisa diferentes aspectos do conteúdo para detectar padrões que podem indicar geração artificial.

> 💡 **Nota**: Este projeto foi desenvolvido com auxílio de IA, seguindo a filosofia de "coding with AI" para criar uma ferramenta útil e educacional.

## ✨ Características

- 🔍 **Análise de Fluxo Óptico**
  - Detecta movimentos inconsistentes ou artificiais
  - Analisa a suavidade das transições
  - Identifica padrões de movimento não naturais

- 👤 **Análise de Consistência Facial**
  - Detecta inconsistências em faces
  - Analisa a estabilidade das características faciais
  - Identifica possíveis manipulações em rostos

- 📊 **Análise de Metadados**
  - Extrai informações técnicas do vídeo
  - Verifica padrões de codificação
  - Analisa propriedades do arquivo

- 🎨 **Análise de Padrões de Ruído**
  - Detecta padrões artificiais de ruído
  - Analisa a distribuição de frequências
  - Identifica características de geração por IA

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/memector.git
cd memector
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

```bash
python memector.py seu_video.mp4 [opções]
```

### Opções Disponíveis:

- `--flow-analysis`: Realiza análise de consistência de fluxo óptico
- `--face-analysis`: Realiza análise de consistência facial
- `--metadata-analysis`: Analisa metadados do vídeo
- `--noise-analysis`: Realiza análise de padrões de ruído
- `--all`: Executa todas as análises disponíveis

### Exemplo:

```bash
python memector.py video_suspeito.mp4 --all
```

## 🛠️ Tecnologias Utilizadas

- Python 3.8+
- OpenCV 4.8+
- NumPy
- Termcolor
- Tqdm

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma Branch para sua Feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2024 Memector

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ⚠️ Aviso Legal

Esta ferramenta é fornecida "como está", sem garantias de qualquer tipo. Os resultados da análise não devem ser considerados como prova definitiva de geração por IA, mas sim como indicadores que podem auxiliar na identificação de conteúdo potencialmente artificial.

## 🌟 Agradecimentos

- Desenvolvido com auxílio de IA
- Inspirado na necessidade de ferramentas de detecção de conteúdo gerado por IA
- Contribuidores da comunidade open source

## 📞 Contato

Para sugestões, bugs ou contribuições, por favor abra uma issue no GitHub.

---

<div align="center">
Made with ❤️ and 🤖
</div> 