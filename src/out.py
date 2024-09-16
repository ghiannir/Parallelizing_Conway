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
    # Non invertiamo le matrici per allinearle correttamente
    hover_text = []
    for i in range(grid.shape[0]):
        hover_text_row = []
        for j in range(grid.shape[1]):
            hover_text_row.append(f"Streak: {streak[i, j]}<br>Counter: {cnt[i, j]}")
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
        title="Final configuration",
        title_font_size=26,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        autosize=False,
        width=1000,
        height=1000,
        plot_bgcolor='#D3D3D3',  # Colore griglia grigio chiaro
        margin=dict(l=10, r=10, t=50, b=10)  # Margini
    )

    # Mostra la figura
    fig.show()

def plot_interactive_gradient_grid(grid, title, colorscale, hover_text):
    """
    Visualizza una griglia con sfumature di colore basate sui valori di grid e aggiunge interattività con hover text.
    """
    # Non invertiamo più la griglia per evitare lo sfasamento
    # Crea la heatmap interattiva con hover text
    fig = go.Figure(data=go.Heatmap(
        z=grid,  # Usa la griglia senza flip
        text=hover_text,
        hoverinfo='text',  # Mostra solo il valore della cella
        colorscale=colorscale,
        showscale=True,
        xgap=2,
        ygap=2
    ))

    fig.update_layout(
        title=title,
        title_font_size=26,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig.show()

# Carica le griglie dai file
grid = np.flipud(load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\mat.txt'))  # Inverti la griglia verticalmente
streak = np.flipud(load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\streak.txt'))  # Inverti la griglia verticalmente
cnt = np.flipud(load_grid(r'C:\Users\mirko\Desktop\Conway\Output .py\cnt.txt'))  # Inverti la griglia verticalmente

# Crea i tooltip interattivi per le griglie di counter e streak (senza invertire le matrici)
hover_text_counter = [[f"Counter: {cnt[i, j]}" for j in range(cnt.shape[1])] for i in range(cnt.shape[0])]
hover_text_streak = [[f"Streak: {streak[i, j]}" for j in range(streak.shape[1])] for i in range(streak.shape[0])]

# Visualizza la griglia originale con tooltip
plot_grid_with_tooltip(grid, streak, cnt)

# Visualizza la griglia interattiva con sfumatura basata su counter
plot_interactive_gradient_grid(cnt, "Number of generations that a cell was alive", 'Viridis', hover_text_counter)

# Visualizza la griglia interattiva con sfumatura basata su streak
plot_interactive_gradient_grid(streak, "Number of generations that each cell remains consecutive alive", 'Cividis', hover_text_streak)
