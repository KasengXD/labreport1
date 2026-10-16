import time
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. GRAPH INITIALIZATION & DEFAULT SETTINGS
# ==============================================================================

GRAPH_DEFAULT = {
    'A': ['B', 'D'],
    'B': ['C', 'E', 'G'],
    'C': ['A'],
    'D': ['C'],
    'E': ['H'],
    'F': [],
    'G': ['F'],
    'H': ['F', 'G']
}

DEFAULT_EDGES = '\n'.join(
    f'{u} {v}' if vs else u
    for u, ns in GRAPH_DEFAULT.items()
    for vs in (ns or [None])
    for v in ([vs] if vs else [None])
).replace(' None', '')

DEFAULTS = {
    'edges': DEFAULT_EDGES,
    'start': 'A',
    'goal': 'No goal',
    'method': 'Both',
    'speed': 0.8
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key in [k for k in st.session_state if k.startswith(('BFS_', 'DFS_'))]:
        del st.session_state[key]


# ==============================================================================
# 2. HELPER FUNCTIONS & SEARCH LOGIC
# ==============================================================================

def normalize(graph):
    result = {str(k).strip().upper(): sorted(set(str(v).strip().upper() for v in vs)) for k, vs in graph.items()}
    for vs in list(result.values()):
        for v in vs:
            result.setdefault(v, [])
    return dict(sorted(result.items()))


def edges_from_text(text):
    """Parses directed edges from text area."""
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


def search(graph, start, goal=None, method='BFS'):
    graph = normalize(graph)
    start = start.upper().strip()
    goal = goal.upper().strip() if goal else None

    if start not in graph or (goal and goal not in graph):
        raise ValueError('Start and goal must exist in the graph.')
    if method not in ('BFS', 'DFS'):
        raise ValueError('Method must be BFS or DFS.')

    frontier = deque([start]) if method == 'BFS' else [start]
    discovered = {start} if method == 'BFS' else set()
    parent = {start: None}
    depth = {start: 0}
    order, trace = [], []

    while frontier:
        node = frontier.popleft() if method == 'BFS' else frontier.pop()

        if method == 'DFS':
            if node in discovered:
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


# ==============================================================================
# 3. GRAPH VISUALIZATION (FIXED EDGE CURVATURE)
# ==============================================================================

def draw_graph(graph_dict, result, step, goal_node=None):
    g = nx.DiGraph()
    for node, neighbors in graph_dict.items():
        g.add_node(node)
        for neighbor in neighbors:
            g.add_edge(node, neighbor)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    try:
        pos = nx.kamada_kawai_layout(g)
    except Exception:
        pos = nx.spring_layout(g, seed=13, k=1.2)

    current_step = result['trace'][step - 1]
    curr_node = current_step['current'].replace(' (skip)', '')
    seen_set = set(current_step['visited'])
    frontier_set = set(current_step['frontier'])

    # Distinct multi-color coding for node states
    node_colors = []
    node_border_colors = []
    
    for n in g.nodes():
        if n == curr_node:
            node_colors.append('#10b981')        # Current node: Vivid Green
            node_border_colors.append('#047857')
        elif n in frontier_set:
            node_colors.append('#f59e0b')        # Frontier node: Amber Orange
            node_border_colors.append('#b45309')
        elif n in seen_set:
            node_colors.append('#93c5fd')        # Visited node: Light Blue
            node_border_colors.append('#1d4ed8')
        else:
            node_colors.append('#e5e7eb')        # Unvisited: Light Slate
            node_border_colors.append('#94a3b8')

    # Draw nodes and labels
    nodes_layer = nx.draw_networkx_nodes(
        g, pos, ax=ax, 
        node_color=node_colors, 
        node_size=1050, 
        linewidths=1.8
    )
    if nodes_layer:
        nodes_layer.set_edgecolor(node_border_colors)

    nx.draw_networkx_labels(g, pos, ax=ax, font_weight='bold', font_size=11)

    # Separate 1-way (straight) edges and 2-way reciprocal (curved) edges
    straight_edges = [(u, v) for u, v in g.edges() if not g.has_edge(v, u)]
    curved_edges = [(u, v) for u, v in g.edges() if g.has_edge(v, u)]

    # 1. Draw 1-way edges as straight lines (rad=0) to avoid bowing into adjacent nodes
    if straight_edges:
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edgelist=straight_edges,
            edge_color='#64748b',
            width=1.3,
            arrows=True,
            arrowsize=18,
            connectionstyle='arc3, rad=0',      # Straight lines for 1-way edges
            min_target_margin=22
        )

    # 2. Draw 2-way reciprocal edges with curves (rad=0.2) so opposing arrows don't overlap
    if curved_edges:
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edgelist=curved_edges,
            edge_color='#64748b',
            width=1.3,
            arrows=True,
            arrowsize=18,
            connectionstyle='arc3, rad=0.2',    # Curve only when reciprocal cycle exists
            min_target_margin=22
        )

    # Highlight target goal node if specified
    if goal_node and goal_node in g.nodes():
        nx.draw_networkx_nodes(
            g, {goal_node: pos[goal_node]}, ax=ax,
            node_color='none',
            node_size=1300,
            linewidths=2.5,
            edgecolors='#ef4444'
        )

    ax.axis('off')
    return fig


# ==============================================================================
# 4. STREAMLIT PAGE LAYOUT
# ==============================================================================

st.set_page_config(page_title='BSD3513 | LAB REPORT 1', page_icon='🔎', layout='wide')

# --- Sidebar ---
with st.sidebar:
    st.header('Graph editor')
    edge_text = st.text_area('Directed edges (one per line: A B)', key='edges', height=260)
    st.caption('A line containing only a node represents an isolated node. Edges are directed.')
    speed = st.slider('Autoplay speed (seconds per step)', 0.2, 2.0, step=0.1, key='speed')
    st.button('Reset to default settings', on_click=reset_defaults, use_container_width=True)

# Parse graph input
try:
    graph = edges_from_text(edge_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# --- Main Content Header ---
st.title('BFS & DFS | Interactive Uninformed Search Algorithms')
st.caption('BSD3513 Artificial Intelligence · Lab Report 1 · WONG KA SENG · Breath First Search & Depth First Search · Alphabetical tie-breaking · UMPSA')

with st.expander('Learning guide', expanded=False):
    st.markdown('**BFS** explores level by level using a FIFO queue; **DFS** explores deeply using a LIFO stack. For adjacency-list implementations both require **O(V + E)** time. BFS finds minimum-hop paths in unweighted graphs; DFS does not guarantee this.')

# --- Control Inputs (3 Columns) ---
nodes = list(graph)
if st.session_state.start not in nodes:
    st.session_state.start = nodes[0]
if st.session_state.goal not in ['No goal'] + nodes:
    st.session_state.goal = 'No goal'

col1, col2, col3 = st.columns(3)
start = col1.selectbox('Starting node', nodes, key='start')
goal_input = col2.selectbox('Goal node (optional)', ['No goal'] + nodes, key='goal')
method = col3.selectbox('Algorithm', ['Both', 'BFS', 'DFS'], key='method')

goal = None if goal_input == 'No goal' else goal_input
results = {m: search(graph, start, goal, m) for m in (['BFS', 'DFS'] if method == 'Both' else [method])}

# --- Results & Visualizers ---
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

    st.caption("💡 Click Autoplay to watch the search algorithm explore each node step by step.")
    play = st.button(f'▶ Autoplay {m}', key=f'play_{m}')

    left, right = st.columns([1.2, 1])
    with left:
        graph_box = st.empty()
        frontier_box = st.empty()
    with right:
        st.write('**Step trace**')
        st.dataframe(pd.DataFrame(r['trace']), hide_index=True, use_container_width=True)

    for s in (range(1, len(r['trace']) + 1) if play else [step]):
        fig = draw_graph(graph, r, s, goal_node=goal)
        graph_box.pyplot(fig)
        plt.close(fig)
        frontier_box.info(
            f'Step {s} of {len(r["trace"])} · current: {r["trace"][s - 1]["current"]} · frontier: {r["trace"][s - 1]["frontier"]}'
        )
        if play:
            time.sleep(speed)

st.divider()

# --- BFS vs DFS Comparison Table ---
if method == 'Both':
    st.subheader('BFS vs DFS comparison')
    bfs, dfs = results['BFS'], results['DFS']
    rows = {
        'Data structure': ('FIFO queue', 'LIFO stack'),
        'Expansion order': (' → '.join(bfs['order']), ' → '.join(dfs['order'])),
        'Expanded nodes': (bfs['explored'], dfs['explored']),
        'Maximum recorded frontier': (bfs['max_frontier'], dfs['max_frontier']),
        'Deepest level reached': (max(bfs['depths'].values(), default=0), max(dfs['depths'].values(), default=0)),
    }
    if goal:
        rows['Goal path'] = (' → '.join(bfs['path']) or 'Not reachable', ' → '.join(dfs['path']) or 'Not reachable')
        rows['Goal path edges'] = (len(bfs['path']) - 1 if bfs['found'] else '—', len(dfs['path']) - 1 if dfs['found'] else '—')

    comparison = pd.DataFrame([(k, str(v[0]), str(v[1])) for k, v in rows.items()], columns=['Measure', 'BFS', 'DFS'])
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    if goal and bfs['found'] and dfs['found'] and len(bfs['path']) < len(dfs['path']):
        st.success(f'BFS found a shorter path to {goal} ({len(bfs["path"]) - 1} edges vs {len(dfs["path"]) - 1}), as expected for unweighted graphs.')
