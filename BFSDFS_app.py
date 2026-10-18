import time
from collections import deque
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

st.set_page_config(page_title='BSD3513 | LAB REPORT 1', page_icon='🔎', layout='wide')

GRAPH = {'A':['B','D'], 'B':['C','E','G'], 'C':['A'], 'D':['C'], 'E':['H'], 'F':[], 'G':['F'], 'H':['F','G']}
DEFAULT_EDGES = '\n'.join(f'{u} {v}' if v else u for u, ns in GRAPH.items() for v in (ns or [None]))
DEFAULTS = {'edges':DEFAULT_EDGES, 'start':'A', 'goal':'No goal', 'method':'Both', 'speed':0.8}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def normalize(graph):
    g = {str(k).strip().upper(): sorted(set(str(v).strip().upper() for v in vs)) for k, vs in graph.items()}
    for ns in list(g.values()):
        for n in ns:
            g.setdefault(n, [])
    return dict(sorted(g.items()))


def search(graph, start, goal=None, method='BFS'):
    graph, start = normalize(graph), start.upper().strip()
    goal = goal.upper().strip() if goal else None
    if start not in graph or (goal and goal not in graph):
        raise ValueError('Start and goal must exist in the graph.')
    if method not in ('BFS', 'DFS'):
        raise ValueError('Method must be BFS or DFS.')

    frontier = deque([start]) if method == 'BFS' else [start]
    discovered = {start} if method == 'BFS' else set()
    parent, depth, order, trace = {start:None}, {start:0}, [], []

    while frontier:
        node = frontier.popleft() if method == 'BFS' else frontier.pop()

        if method == 'DFS':
            if node in discovered:
                trace.append({'step':len(trace)+1, 'current':f'{node} (skip)', 'visited':order.copy(),
                              'frontier':list(frontier), 'depth':depth[node], 'parent':parent[node]})
                continue
            discovered.add(node)

        order.append(node)
        if node != goal:
            neighbors = graph[node] if method == 'BFS' else list(reversed(graph[node]))
            for nxt in neighbors:
                if nxt not in discovered:
                    if method == 'BFS':
                        discovered.add(nxt)
                    parent[nxt], depth[nxt] = node, depth[node]+1
                    frontier.append(nxt)

        trace.append({'step':len(trace)+1, 'current':node, 'visited':order.copy(),
                      'frontier':list(frontier), 'depth':depth[node], 'parent':parent[node]})
        if node == goal:
            break

    path = []
    if goal is not None and goal in order:
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

    return {'method':method, 'order':order, 'path':path, 'found':bool(path) if goal else None,
            'trace':trace, 'depths':{n:depth[n] for n in order},
            'parents':{n:parent[n] for n in order}, 'explored':len(order),
            'max_frontier':max([1]+[len(t['frontier']) for t in trace])}


def edges_from_text(text):
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


def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for k in [k for k in st.session_state if k.startswith(('BFS_', 'DFS_', 'play_'))]:
        del st.session_state[k]


def change_step(key, delta, max_step):
    st.session_state[key] = max(1, min(max_step, st.session_state.get(key, 1) + delta))


# Improved graph visualization
def draw_graph(result, step):
    g = nx.DiGraph()
    g.add_nodes_from(graph)
    g.add_edges_from((u, v) for u, ns in graph.items() for v in ns)

    default_edges = {(u, v) for u, ns in GRAPH.items() for v in ns}
    if set(g.nodes) == set(GRAPH) and set(g.edges) == default_edges:
        pos = {'A':(0,0), 'B':(-1,1), 'C':(1,-0.2), 'D':(2,-0.65),
               'E':(-2,1.5), 'H':(-1.5,2.6), 'G':(0,2), 'F':(0.8,3)}
    else:
        pos = nx.spring_layout(g, seed=13)

    item = result['trace'][step-1]
    visited = set(item['visited'])
    frontier = set(item['frontier']) - visited
    raw = item['current']
    current = raw.replace(' (skip)', '')
    skip = raw.endswith('(skip)')

    colors = ['#10b981' if n == current and not skip else '#93c5fd' if n in visited
              else '#fbbf24' if n in frontier else '#e5e7eb' for n in g.nodes]
    borders = ['#dc2626' if goal and n == goal else '#7c3aed' if n == start else '#ffffff' for n in g.nodes]

    tree = [(p, c) for c, p in result['parents'].items()
            if p is not None and p in visited and c in visited]
    path = list(zip(result['path'], result['path'][1:])) if goal and result['found'] and goal in visited else []

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='#cbd5e1', width=1.4,
                           arrows=True, arrowsize=18, node_size=850)
    if tree:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=tree, edge_color='#0f766e',
                               width=2.4, arrows=True, arrowsize=20, node_size=850)
    if path:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=path, edge_color='#f59e0b',
                               width=3.8, arrows=True, arrowsize=23, node_size=850)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, edgecolors=borders,
                           linewidths=2.3, node_size=850)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=11, font_weight='bold')
    ax.set_title(f"{result['method']} Search | Step {step}/{len(result['trace'])} | Current: {current}",
                 fontsize=12, fontweight='bold')

    handles = [Line2D([0],[0], marker='o', color='w', label=label,
                      markerfacecolor=color, markersize=8)
               for label, color in [('Current','#10b981'), ('Expanded','#93c5fd'),
                                    ('Frontier','#fbbf24'), ('Undiscovered','#e5e7eb')]]
    handles.append(Line2D([0],[0], color='#0f766e', lw=2.4, label='Search tree'))
    if goal:
        handles.append(Line2D([0],[0], color='#f59e0b', lw=3.8, label='Goal path'))

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5,-0.04),
              ncol=3, frameon=False, fontsize=8)
    ax.axis('off')
    fig.tight_layout(rect=[0,0.06,1,1])
    return fig


# Main interface
st.title('BFS & DFS | Interactive Uninformed Search Algorithms')
st.caption('BSD3513 Artificial Intelligence · Lab Report 1 · Wong Ka Seng · Breadth-First Search & Depth-First Search · UMPSA')

with st.expander('Learning guide'):
    st.markdown('**BFS** explores level by level using a FIFO queue; **DFS** explores deeply using a LIFO stack. Both take **O(V + E)** time with adjacency lists. BFS finds minimum-hop paths in unweighted graphs; DFS does not guarantee this.')

with st.sidebar:
    st.header('Graph editor')
    edge_text = st.text_area('Directed edges (one per line: A B)', key='edges', height=220)
    st.caption('A single node per line creates an isolated node. Edges are directed.')
    st.slider('Autoplay speed (seconds per step)', 0.2, 2.0, step=0.1, key='speed')
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
goal_choice = b.selectbox('Goal node (optional)', ['No goal'] + nodes, key='goal')
method = c.selectbox('Algorithm', ['Both', 'BFS', 'DFS'], key='method')
goal = None if goal_choice == 'No goal' else goal_choice

results = {m:search(graph, start, goal, m) for m in (['BFS','DFS'] if method == 'Both' else [method])}

for m, r in results.items():
    st.subheader(f'{m} search')
    st.write('**Expansion order:** ' + ' → '.join(r['order']))
    if goal:
        st.write('**Goal path:** ' + (' → '.join(r['path']) if r['found'] else 'Not reachable'))

    x, y, z = st.columns(3)
    x.metric('Expanded nodes', r['explored'])
    y.metric('Max frontier', r['max_frontier'])
    z.metric('Goal path edges', len(r['path'])-1 if r['found'] else '—')

    # Button-based step selection instead of a slider
    step_key = f'{m}_step'
    st.session_state.setdefault(step_key, 1)
    st.session_state[step_key] = max(1, min(st.session_state[step_key], len(r['trace'])))
    current_step = st.session_state[step_key]

    prev_col, info_col, next_col = st.columns([1, 1.25, 1])
    with prev_col:
        st.button('◀ Previous', key=f'{m}_prev', disabled=current_step <= 1,
                  on_click=change_step, args=(step_key, -1, len(r['trace'])), width='stretch')
    with info_col:
        st.markdown(
            f'<div style="text-align:center;padding-top:7px;font-weight:600">'
            f'Step {current_step} of {len(r["trace"])}</div>',
            unsafe_allow_html=True
        )
    with next_col:
        st.button('Next ▶', key=f'{m}_next', disabled=current_step >= len(r['trace']),
                  on_click=change_step, args=(step_key, 1, len(r['trace'])), width='stretch')

    step = st.session_state[step_key]
    play = st.button(f'▶ Autoplay {m}', key=f'play_{m}', width='stretch')
    left, right = st.columns([1.2, 1])

    with left:
        graph_box, frontier_box = st.empty(), st.empty()
    with right:
        st.write('**Step trace**')
        st.dataframe(pd.DataFrame(r['trace']), hide_index=True, width='stretch')

    for s in (range(1, len(r['trace'])+1) if play else [step]):
        fig = draw_graph(r, s)
        graph_box.pyplot(fig, use_container_width=True)
        plt.close(fig)

        t = r['trace'][s-1]
        frontier_text = ' → '.join(t['frontier']) if t['frontier'] else 'Empty'
        structure = 'FIFO queue' if m == 'BFS' else 'LIFO stack'
        frontier_box.info(
            f"Step {s}/{len(r['trace'])} · Current: {t['current']} · Depth: {t['depth']}\n\n"
            f"**Frontier ({structure}):** {frontier_text}"
        )
        if play:
            st.session_state[step_key] = s
            time.sleep(st.session_state.speed)

st.divider()
if method == 'Both':
    st.subheader('BFS vs DFS comparison')
    bfs, dfs = results['BFS'], results['DFS']
    rows = {
        'Data structure':('FIFO queue','LIFO stack'),
        'Expansion order':(' → '.join(bfs['order']),' → '.join(dfs['order'])),
        'Expanded nodes':(bfs['explored'],dfs['explored']),
        'Max frontier':(bfs['max_frontier'],dfs['max_frontier']),
        'Deepest level':(max(bfs['depths'].values()),max(dfs['depths'].values()))
    }
    if goal:
        rows['Goal path'] = (' → '.join(bfs['path']) or 'Not reachable',
                             ' → '.join(dfs['path']) or 'Not reachable')
        rows['Goal path edges'] = (len(bfs['path'])-1 if bfs['found'] else '—',
                                   len(dfs['path'])-1 if dfs['found'] else '—')

    st.dataframe(pd.DataFrame([(k,str(v[0]),str(v[1])) for k,v in rows.items()],
                              columns=['Measure','BFS','DFS']), hide_index=True, width='stretch')

    if goal and bfs['found'] and dfs['found'] and len(bfs['path']) < len(dfs['path']):
        st.success(f"BFS found a shorter path to {goal} ({len(bfs['path'])-1} edges vs {len(dfs['path'])-1}), as expected for unweighted graphs.")
