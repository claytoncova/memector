import cv2
import numpy as np
import argparse
from termcolor import colored, cprint
from tqdm import tqdm
import os
import sys

# --- Configurações e Utilitários ---
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Carregar o detector de faces do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def print_header(title):
    """Imprime um cabeçalho colorido para a seção."""
    print("\n" + "="*80)
    cprint(f"--- {title} ---", 'cyan', attrs=['bold'])
    print("="*80 + "\n")

def print_section_title(title):
    """Imprime um título de seção."""
    cprint(f"\n>>> {title}\n", 'yellow', attrs=['bold'])

def print_result(label, value, color='green'):
    """Imprime um resultado formatado."""
    cprint(f"    {label:<30}: {str(value)}", color)

def print_error(message):
    """Imprime uma mensagem de erro."""
    cprint(f"ERRO: {message}", 'red', attrs=['bold'])

def print_warning(message):
    """Imprime uma mensagem de aviso."""
    cprint(f"AVISO: {message}", 'yellow')

# --- Funções de Análise ---

def analyze_optical_flow_consistency(video_path):
    """
    Analisa a consistência do fluxo óptico no vídeo.
    Retorna uma métrica de anomalia do fluxo óptico.
    """
    print_section_title("Análise de Consistência de Fluxo Óptico")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_error(f"Não foi possível abrir o vídeo: {video_path}")
        return None

    ret, old_frame = cap.read()
    if not ret:
        print_error("Não foi possível ler o primeiro frame do vídeo.")
        cap.release()
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    flow_deviation_magnitudes = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print_warning("Esta análise pode ser demorada para vídeos longos. Aguarde...")

    # Usando tqdm para barra de progresso
    for _ in tqdm(range(total_frames - 1), desc="Processando Fluxo Óptico"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if p0 is None or len(p0) == 0: # Re-detect points if lost
            p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if p0 is None or len(p0) == 0: # If still no points, skip
                continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **LK_PARAMS)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0 and len(good_old) > 0:
                # Calcula o vetor de movimento
                flow_vectors = good_new - good_old
                # Calcula a magnitude do vetor de movimento
                magnitudes = np.linalg.norm(flow_vectors, axis=1)
                # Adiciona a média das magnitudes para análise
                flow_deviation_magnitudes.append(np.mean(magnitudes))

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if p1 is not None and len(p1[st==1]) > 0 else None

    cap.release()

    if not flow_deviation_magnitudes:
        print_warning("Não foi possível coletar dados de fluxo óptico suficientes.")
        return 0.0

    # Uma métrica simples de anomalia: desvio padrão das magnitudes do fluxo.
    # Valores muito baixos ou muito altos podem indicar padrões artificiais.
    anomaly_score = np.std(flow_deviation_magnitudes)
    print_result("Desvio Padrão do Fluxo Óptico", f"{anomaly_score:.4f}")
    if anomaly_score < 1.0: # Limiar arbitrário, pode ser ajustado
        print_warning("Desvio padrão de fluxo óptico muito baixo, pode indicar movimento 'perfeito' ou falta de ruído natural.")
    elif anomaly_score > 10.0: # Limiar arbitrário
        print_warning("Desvio padrão de fluxo óptico muito alto, pode indicar movimentos abruptos ou inconsistentes.")
    else:
        print_result("Status do Fluxo Óptico", "Consistente", 'green')

    return anomaly_score

def analyze_face_consistency(video_path):
    """
    Analisa a consistência facial no vídeo usando OpenCV Cascade Classifier.
    Retorna uma contagem de inconsistências detectadas.
    """
    print_section_title("Análise de Consistência Facial")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_error(f"Não foi possível abrir o vídeo: {video_path}")
        return None

    inconsistency_count = 0
    first_face_features = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print_warning("Esta análise pode ser demorada para vídeos longos. Aguarde...")

    # Usando tqdm para barra de progresso
    for frame_idx in tqdm(range(total_frames), desc="Processando Consistência Facial"):
        ret, frame = cap.read()
        if not ret:
            break

        # Converte para escala de cinza para detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta faces usando o Cascade Classifier
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            continue

        # Para cada face detectada
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue

            try:
                # Redimensiona para um tamanho fixo para comparação
                face_features = cv2.resize(face_roi, (100, 100))
                
                if first_face_features is None:
                    first_face_features = face_features
                else:
                    # Calcula o histograma da face atual
                    hist_current = cv2.calcHist([face_features], [0], None, [256], [0, 256])
                    cv2.normalize(hist_current, hist_current, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    # Calcula o histograma da primeira face
                    hist_first = cv2.calcHist([first_face_features], [0], None, [256], [0, 256])
                    cv2.normalize(hist_first, hist_first, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    # Compara os histogramas usando correlação
                    similarity = cv2.compareHist(hist_current, hist_first, cv2.HISTCMP_CORREL)
                    
                    # Se a similaridade for muito baixa, considera como inconsistência
                    if similarity < 0.6:  # Limiar ajustado para ser mais tolerante
                        inconsistency_count += 1
                        break  # Uma inconsistência por frame é suficiente
            except cv2.error as e:
                print_warning(f"Erro ao processar face no frame {frame_idx}: {str(e)}")
                continue

    cap.release()

    print_result("Faces Inconsistentes Detectadas", inconsistency_count, 'red' if inconsistency_count > 0 else 'green')
    if inconsistency_count > 0:
        print_warning("Múltiplas inconsistências faciais podem indicar geração por IA ou edição profunda.")
    else:
        print_result("Status da Consistência Facial", "Consistente", 'green')

    return inconsistency_count

def analyze_metadata(video_path):
    """
    Analisa metadados básicos do vídeo.
    Esta função é um placeholder, metadados detalhados exigem libs como `hachoir-parser` ou `mediainfo`.
    """
    print_section_title("Análise de Metadados Básicos")
    # A biblioteca `cv2` fornece metadados básicos, mas não informações de software de autoria
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_error(f"Não foi possível abrir o vídeo: {video_path}")
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0

    cap.release()

    print_result("Caminho do Arquivo", os.path.basename(video_path))
    print_result("Resolução", f"{width}x{height}")
    print_result("FPS (Frames por Segundo)", f"{fps:.2f}")
    print_result("Número de Frames", frame_count)
    print_result("Duração (segundos)", f"{duration_sec:.2f}")

    # Indícios de IA baseados em metadados são limitados sem ferramentas mais avançadas
    # mas formatos incomuns ou falta de certas informações podem ser um sinal.
    if fps == 0 or frame_count == 0:
        print_warning("Metadados incompletos ou vídeo inválido.")
        return {"status": "incompleto"}

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec
    }

def analyze_noise_patterns(video_path):
    """
    Analisa padrões de ruído (conceitual, exige FFT/DCT mais aprofundados).
    Esta função é um placeholder para uma análise de espectro mais complexa.
    """
    print_section_title("Análise de Padrões de Ruído (Conceitual)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_error(f"Não foi possível abrir o vídeo: {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print_error("Não foi possível ler o primeiro frame para análise de ruído.")
        cap.release()
        return None

    # Converte para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Converte para float32 para FFT
    f = np.fft.fft2(gray_frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-10) # Adiciona um epsilon para evitar log(0)

    # Para uma análise real, precisaríamos analisar a distribuição do espectro,
    # procurar por picos anômalos ou ausência de ruído de alta frequência.
    # Aqui, apenas calculamos a média para demonstração.
    mean_magnitude = np.mean(magnitude_spectrum)
    print_result("Média do Espectro de Magnitude", f"{mean_magnitude:.4f}")

    # Um valor muito baixo ou muito "perfeito" (com desvio padrão muito baixo)
    # nas frequências altas pode ser um indício de IA.
    # Isto é uma simplificação. Uma análise mais aprofundada envolveria:
    # 1. Análise de componentes de frequência específicas (e.g., JPEG grid artifacts)
    # 2. Comparação com perfis de ruído de câmeras reais
    # 3. Métricas como PRNU (Photo-Response Non-Uniformity)
    print_warning("Esta é uma análise conceitual de ruído. Requer algoritmos avançados para resultados concretos.")

    cap.release()
    return {"mean_magnitude_spectrum": mean_magnitude}

# --- Função Principal ---

def main():
    parser = argparse.ArgumentParser(
        description=colored("AI Video Detector - Ferramenta para identificar indícios de IA em vídeos.", 'green', attrs=['bold']),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video_path", help="Caminho para o arquivo de vídeo a ser analisado.")
    parser.add_argument("--flow-analysis", action="store_true",
                        help="Realiza análise de consistência de fluxo óptico.")
    parser.add_argument("--face-analysis", action="store_true",
                        help="Realiza análise de consistência facial usando OpenCV.")
    parser.add_argument("--metadata-analysis", action="store_true",
                        help="Analisa metadados básicos do vídeo.")
    parser.add_argument("--noise-analysis", action="store_true",
                        help="Realiza análise conceitual de padrões de ruído (FFT/DCT).")
    parser.add_argument("--all", action="store_true",
                        help="Executa todas as análises disponíveis.")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print_error(f"O arquivo de vídeo '{args.video_path}' não foi encontrado.")
        sys.exit(1)

    cprint(r"""
███╗   ███╗███████╗███╗   ███╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
████╗ ████║██╔════╝████╗ ████║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██╔████╔██║█████╗  ██╔████╔██║█████╗  ██║        ██║   ██║   ██║██████╔╝
██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                                                                        
  """, 'magenta', attrs=['bold'])
    cprint("AI Video Detector", 'magenta', attrs=['bold'])
    cprint("Ferramenta para identificar indícios de IA em vídeos\n", 'magenta')

    print_header("Início da Análise de Vídeo")
    print_result("Vídeo Sendo Analisado", args.video_path)

    any_analysis_performed = False

    if args.flow_analysis or args.all:
        analyze_optical_flow_consistency(args.video_path)
        any_analysis_performed = True

    if args.face_analysis or args.all:
        analyze_face_consistency(args.video_path)
        any_analysis_performed = True

    if args.metadata_analysis or args.all:
        analyze_metadata(args.video_path)
        any_analysis_performed = True

    if args.noise_analysis or args.all:
        analyze_noise_patterns(args.video_path)
        any_analysis_performed = True

    if not any_analysis_performed:
        print_warning("Nenhuma análise especificada. Use --help para ver as opções ou --all para executar todas.")

    print_header("Análise Concluída")
    cprint("Lembre-se: A detecção de IA é complexa. Múltiplos indícios fornecem maior confiança.", 'white', attrs=['bold'])

if __name__ == "__main__":
    main()