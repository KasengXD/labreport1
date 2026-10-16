import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import deque

GRAPH = {'A':['B','D'], 'B':['C','E','G'], 'C':['A'], 'D':['C'], 'E':['H'], 'F':[], 'G':['F'], 'H':['F','G']}


def normalize(graph):
    result = {str(k).strip().upper(): sorted(set(str(v).strip().upper() for v in vs)) for k, vs in graph.items()}
    for vs in list(result.values()):
        for v in vs:
            result.setdefault(v, [])
    return dict(sorted(result.items()))


def search(graph, start, goal=None, method='BFS'):
    graph = normalize(graph)
    start = start.upper().strip()
    goal = goal.upper().strip() if goal else None
    if start not in graph or (goal and goal not in graph):
        raise ValueError('Start and goal must exist in the graph.')
    if method not in ('BFS', 'DFS'):
        raise ValueError('Method must be BFS or DFS.')

    frontier = deque([start]) if method == 'BFS' else [start]
    # BFS marks a node when it is enqueued; DFS marks a node when it is expanded,
    # so the deepest branch reaches a node first (same result as recursive DFS).
    discovered = {start} if method == 'BFS' else set()
    parent = {start: None}
    depth = {start: 0}
    order, trace = [], []

    while frontier:
        node = frontier.popleft() if method == 'BFS' else frontier.pop()

        if method == 'DFS':
            if node in discovered:  # duplicate stack entry: already expanded, skip it
                trace.append({
                    'step': len(trace) + 1,
                    'current': f'{node} (skip)',
                    'visited': order.copy(),
                    'frontier': list(frontier),
                    'depth': depth[node],
                    'parent': parent[node],
                })
                continue
            discovered.add(node)

        order.append(node)

        if node != goal:
            neighbors = graph[node] if method == 'BFS' else list(reversed(graph[node]))
            for neighbor in neighbors:
                if neighbor not in discovered:
                    if method == 'BFS':
                        discovered.add(neighbor)
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    frontier.append(neighbor)

        trace.append({
            'step': len(trace) + 1,
            'current': node,
            'visited': order.copy(),
            'frontier': list(frontier),
            'depth': depth[node],
            'parent': parent[node],
        })

        if node == goal:
            break

    path = []
    if goal is not None and goal in order:
        cursor = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()

    return {
        'method': method,
        'order': order,
        'path': path,
        'found': bool(path) if goal else None,
        'trace': trace,
        'depths': {n: depth[n] for n in order},
        'parents': {n: parent[n] for n in order},
        'explored': len(order),
        'max_frontier': max([1] + [len(t['frontier']) for t in trace]),
    }


def edges_from_text(text):
    """One directed edge per line: A B or A,B; isolated nodes: A."""
    graph = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip().replace(',', ' ')
        if not line:
            continue
        tokens = line.upper().split()
        if len(tokens) not in (1, 2):
            raise ValueError(f'Invalid line: {line}')
        graph.setdefault(tokens[0], [])
        if len(tokens) == 2:
            graph[tokens[0]].append(tokens[1])
            graph.setdefault(tokens[1], [])
    if not graph:
        raise ValueError('Graph cannot be empty.')
    if len(graph) > 100:
        raise ValueError('Maximum 100 nodes.')
    return normalize(graph)


DEFAULT_EDGES = '\n'.join(
    f'{u} {v}' if vs else u
    for u, ns in GRAPH.items()
    for vs in (ns or [None])
    for v in ([vs] if vs else [None])
).replace(' None', '')

# Added feature 1: reset button - default settings kept in session state
DEFAULTS = {'edges': DEFAULT_EDGES, 'start': 'A', 'goal': 'No goal', 'method': 'Both', 'speed': 0.8}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key in [k for k in st.session_state if k.startswith(('BFS_', 'DFS_'))]:
        del st.session_state[key]  # step sliders go back to the last step


st.set_page_config(page_title='BSD3513 | LAB REPORT 1', page_icon='🔎', layout='wide')
st.title('BFS & DFS | Interactive Uninformed Search Algorithms')
st.caption('BSD3513 Artificial Intelligence · Lab Report 1 · Mok Ker Qin · Breath First Search  & Depth First Search · Alphabetical tie-breaking · UMPSA')

with st.expander('Learning guide', expanded=False):
    st.markdown('**BFS** explores level by level using a FIFO queue; **DFS** explores deeply using a LIFO stack. For adjacency-list implementations both require **O(V + E)** time. BFS finds minimum-hop paths in unweighted graphs; DFS does not guarantee this.')

with st.sidebar:
    st.header('Graph editor')
    edge_text = st.text_area('Directed edges (one per line: A B)', key='edges', height=260)
    st.caption('A line containing only a node represents an isolated node. Edges are directed.')
    speed = st.slider('Autoplay speed (seconds per step)', 0.2, 2.0, step=0.1, key='speed')
    st.button('Reset to default settings', on_click=reset_defaults, width='stretch')

try:
    graph = edges_from_text(edge_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

nodes = list(graph)
if st.session_state.start not in nodes:
    st.session_state.start = nodes[0]
if st.session_state.goal not in ['No goal'] + nodes:
    st.session_state.goal = 'No goal'

a, b, c = st.columns(3)
start = a.selectbox('Starting node', nodes, key='start')
goal = b.selectbox('Goal node (optional)', ['No goal'] + nodes, key='goal')
method = c.selectbox('Algorithm', ['Both', 'BFS', 'DFS'], key='method')
goal = None if goal == 'No goal' else goal
results = {m: search(graph, start, goal, m) for m in (['BFS', 'DFS'] if method == 'Both' else [method])}


def draw_graph(result, step):
    g = nx.DiGraph()
    for node, neighbors in graph.items():
        g.add_node(node)
        for neighbor in neighbors:
            g.add_edge(node, neighbor)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = nx.spring_layout(g, seed=13)
    seen = result['trace'][step - 1]['visited']
    colors = ['#10b981' if n == result['trace'][step - 1]['current'] else '#93c5fd' if n in seen else '#e5e7eb' for n in g.nodes]
    nx.draw_networkx(g, pos, ax=ax, with_labels=True, node_color=colors, node_size=950, font_weight='bold', arrows=True, arrowsize=19, edge_color='#64748b')
    ax.axis('off')
    return fig


for m, r in results.items():
    st.subheader(f'{m} search')
    st.write('**Expansion order:** ' + ' → '.join(r['order']))
    if goal:
        st.write('**Goal path:** ' + (' → '.join(r['path']) if r['found'] else 'Not reachable'))
    x, y, z = st.columns(3)
    x.metric('Expanded nodes', r['explored'])
    y.metric('Maximum recorded frontier', r['max_frontier'])
    z.metric('Goal path edges', len(r['path']) - 1 if r['found'] else '—')
    step = st.slider(f'{m}: inspect expansion step', 1, len(r['trace']), len(r['trace']), key=f"{m}_{len(r['trace'])}")

    # Autoplay instruction
    st.caption("💡 Click Autoplay to watch the search algorithm explore each node step by step.")

    play = st.button(f'▶ Autoplay {m}', key=f'play_{m}')  # Added feature 2: autoplay
    left, right = st.columns([1.2, 1])
    with left:
        graph_box = st.empty()
        frontier_box = st.empty()
    with right:
        st.write('**Step trace**')
        st.dataframe(pd.DataFrame(r['trace']), hide_index=True, width='stretch')

    for s in (range(1, len(r['trace']) + 1) if play else [step]):
        fig = draw_graph(r, s)
        graph_box.pyplot(fig)
        plt.close(fig)
        frontier_box.info(f'Step {s} of {len(r["trace"])} · current: {r["trace"][s - 1]["current"]} · frontier: {r["trace"][s - 1]["frontier"]}')
        if play:
            time.sleep(speed)

st.divider()

# Added feature 3: BFS vs DFS comparison table
if method == 'Both':
    st.subheader('BFS vs DFS comparison')
    bfs, dfs = results['BFS'], results['DFS']
    rows = {
        'Data structure': ('FIFO queue', 'LIFO stack'),
        'Expansion order': (' → '.join(bfs['order']), ' → '.join(dfs['order'])),
        'Expanded nodes': (bfs['explored'], dfs['explored']),
        'Maximum recorded frontier': (bfs['max_frontier'], dfs['max_frontier']),
        'Deepest level reached': (max(bfs['depths'].values()), max(dfs['depths'].values())),
    }
    if goal:
        rows['Goal path'] = (' → '.join(bfs['path']) or 'Not reachable', ' → '.join(dfs['path']) or 'Not reachable')
        rows['Goal path edges'] = (len(bfs['path']) - 1 if bfs['found'] else '—', len(dfs['path']) - 1 if dfs['found'] else '—')
    comparison = pd.DataFrame([(k, str(v[0]), str(v[1])) for k, v in rows.items()], columns=['Measure', 'BFS', 'DFS'])
    st.dataframe(comparison, hide_index=True, width='stretch')
    if goal and bfs['found'] and dfs['found'] and len(bfs['path']) < len(dfs['path']):
        st.success(f'BFS found a shorter path to {goal} ({len(bfs["path"]) - 1} edges vs {len(dfs["path"]) - 1}), as expected for unweighted graphs.')
