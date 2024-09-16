import numpy as np
import plotly.graph_objects as go

def load_grid(file_path):
    """
    Carica la griglia da un file.
    Il file deve avere 0 e 1 separati da spazi per indicare lo stato delle celle.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        grid = [list(map(int, line.split())) for line in lines]
    return np.array(grid)

def plot_grid_with_tooltip(grid, streak, cnt):
    """
    Visualizza la griglia usando Plotly e mostra un tooltip con i valori
    di streak e cnt quando il mouse si sposta sopra una cella.
    """
    # Inverti l'asse Y di streak e cnt per allinearli alla visualizzazione Plotly
    streak_flipped = np.flipud(streak)
    cnt_flipped = np.flipud(cnt)

    # Crea i dati da visualizzare in una heatmap con Plotly
    hover_text = []
    for i in range(grid.shape[0]):
        hover_text_row = []
        for j in range(grid.shape[1]):
            hover_text_row.append(f"Streak: {streak_flipped[i, j]}<br>Count: {cnt_flipped[i, j]}")
        hover_text.append(hover_text_row)

    # Crea una heatmap con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        text=hover_text,
        hoverinfo='text',  # Mostra il testo personalizzato (hover_text) al passaggio del mouse
        colorscale=[[0, '#7E7E7E'], [1, '#FBFF2D']],  # Colore per 0 (grigio scuro) e 1 (giallo)
        showscale=False,  # Nascondi la barra dei colori
        xgap=2,  # Distanza orizzontale tra le celle per simulare la griglia
        ygap=2   # Distanza verticale tra le celle per simulare la griglia
    ))

    # Configura il layout
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        autosize=False,
        width=600,
        height=600,
        plot_bgcolor='#D3D3D3',  # Colore griglia grigio chiaro
        margin=dict(l=10, r=10, t=50, b=10)  # Margini
    )

    # Mostra la figura
    fig.show()

# Carica la griglia dal file mat.txt
grid = load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\mat.txt')


# Carica streak.txt e cnt.txt
streak = load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\streak.txt')
cnt = load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\cnt.txt')

# Visualizza la griglia con tooltip
plot_grid_with_tooltip(grid, streak, cnt)
