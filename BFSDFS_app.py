import time
from collections import deque
from html import escape
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
        before = list(frontier)
        node = frontier.popleft() if method == 'BFS' else frontier.pop()

        if method == 'DFS' and node in discovered:
            trace.append({'step':len(trace)+1, 'current':f'{node} (skip)',
                          'visited':order.copy(), 'frontier_before':before,
                          'frontier':list(frontier), 'depth':depth[node],
                          'parent':parent[node], 'neighbors':[], 'added':[],
                          'ignored':[], 'skipped':True})
            continue

        if method == 'DFS':
            discovered.add(node)

        order.append(node)
        neighbors = [] if node == goal else (
            graph[node] if method == 'BFS' else list(reversed(graph[node]))
        )
        added, ignored = [], []

        for nxt in neighbors:
            if nxt not in discovered:
                if method == 'BFS':
                    discovered.add(nxt)
                if nxt not in parent:
                    parent[nxt], depth[nxt] = node, depth[node]+1
                frontier.append(nxt)
                added.append(nxt)
            else:
                ignored.append(nxt)

        trace.append({'step':len(trace)+1, 'current':node, 'visited':order.copy(),
                      'frontier_before':before, 'frontier':list(frontier),
                      'depth':depth[node], 'parent':parent[node],
                      'neighbors':neighbors, 'added':added,
                      'ignored':ignored, 'skipped':False})
        if node == goal:
            break

    path = []
    if goal is not None and goal in order:
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

    return {'method':method, 'order':order, 'path':path,
            'found':bool(path) if goal else None, 'trace':trace,
            'depths':depth.copy(), 'parents':parent.copy(),
            'explored':len(order),
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
    st.session_state[key] = max(1, min(max_step, st.session_state.get(key, 1)+delta))


# Feature 2: Explain each search step.
def explain_step(result, step, goal):
    item = result['trace'][step-1]
    node = item['current'].replace(' (skip)', '')

    if item.get('skipped'):
        return f'**{node} is skipped.** This duplicate DFS stack entry was already expanded, so its neighbours are not checked.'

    if goal and node == goal:
        return f'**Goal reached: {node}.** The algorithm stops without expanding this node’s outgoing edges.'

    method = result['method']
    action = 'dequeued from the front of the queue' if method == 'BFS' else 'popped from the top of the stack'
    parts = [f'**Expand {node}.** It was {action}. Its depth is **{item["depth"]}**.']

    if item['neighbors']:
        parts.append('Neighbours checked: '+', '.join(item['neighbors'])+'.')
    else:
        parts.append('This node has no outgoing neighbours.')

    if item['added']:
        verb = 'enqueued' if method == 'BFS' else 'pushed onto the stack'
        parts.append('Added: **'+', '.join(item['added'])+f'** ({verb}).')

    if item['ignored']:
        parts.append('Not added because already discovered/expanded: **'+', '.join(item['ignored'])+'**.')

    parts.append('The frontier below shows the updated queue or stack.')
    return ' '.join(parts)


# Feature 1: Display the queue or stack as visual boxes.
def frontier_html(method, frontier):
    title = 'BFS queue · front → back' if method == 'BFS' else 'DFS stack · bottom → top (next pop at right)'

    if not frontier:
        cards = '<span style="color:#64748b;font-style:italic">Empty</span>'
    else:
        cards = ''.join(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'min-width:34px;padding:8px 11px;margin:3px;border-radius:8px;'
            f'background:{"#dbeafe" if method=="BFS" else "#fef3c7"};'
            f'border:1px solid {"#60a5fa" if method=="BFS" else "#f59e0b"};'
            f'font-weight:700;color:#172033">{escape(str(n))}</span>'
            for n in frontier
        )

    return (f'<div style="padding:10px 12px;margin:4px 0 10px;border:1px solid #cbd5e1;border-radius:10px">'
            f'<div style="font-weight:650;margin-bottom:7px">{title}</div>'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center">{cards}</div></div>')


# Main graph: coloured search states and highlighted edges.
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
    frontier = set(item['frontier'])-visited
    raw = item['current']
    current = raw.replace(' (skip)', '')
    skip = raw.endswith('(skip)')

    colors = ['#10b981' if n == current and not skip else
              '#93c5fd' if n in visited else
              '#fbbf24' if n in frontier else '#e5e7eb' for n in g.nodes]
    borders = ['#dc2626' if goal and n == goal else
               '#7c3aed' if n == start else '#ffffff' for n in g.nodes]

    tree = [(p, c) for c, p in result['parents'].items()
            if p is not None and p in visited and c in visited]
    path = list(zip(result['path'], result['path'][1:])) if goal and result['found'] and goal in visited else []

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=120)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='#cbd5e1',
                           width=1.4, arrows=True, arrowsize=18, node_size=850)
    if tree:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=tree,
                               edge_color='#0f766e', width=2.4,
                               arrows=True, arrowsize=20, node_size=850)
    if path:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=path,
                               edge_color='#f59e0b', width=3.8,
                               arrows=True, arrowsize=23, node_size=850)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors,
                           edgecolors=borders, linewidths=2.3, node_size=850)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=11, font_weight='bold')
    ax.set_title(f'{result["method"]} Search | Step {step}/{len(result["trace"])} | Current: {current}',
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


# Feature 5: Draw parent-child relationships as a search tree.
def draw_search_tree(result, goal):
    parents, depths = result['parents'], result['depths']
    tree = nx.DiGraph()
    tree.add_nodes_from(parents)
    tree.add_edges_from((p, n) for n, p in parents.items() if p is not None and p in parents)
    if not tree.nodes:
        return None

    children = {n:[] for n in tree.nodes}
    for p, c in tree.edges:
        children[p].append(c)
    for n in children:
        children[n].sort()

    pos, cursor = {}, [0]

    def place(node, level=0):
        kids = children[node]
        if kids:
            x = sum(place(k, level+1) for k in kids)/len(kids)
        else:
            x = cursor[0]
            cursor[0] += 1
        pos[node] = (x, -level)
        return x

    root = next((n for n, p in parents.items() if p is None), next(iter(tree.nodes)))
    place(root)
    visited = set(result['order'])
    node_colors = ['#fca5a5' if goal and n == goal and n in visited else
                   '#93c5fd' if n in visited else '#fbbf24' for n in tree.nodes]
    path_edges = set(zip(result['path'], result['path'][1:])) if goal and result['found'] else set()
    edge_colors = ['#f59e0b' if (u,v) in path_edges else '#0f766e' for u,v in tree.edges]
    edge_widths = [3.2 if (u,v) in path_edges else 2 for u,v in tree.edges]

    fig, ax = plt.subplots(figsize=(8, max(3.2, min(6, 2.2+len(tree.nodes)*0.12))), dpi=110)
    nx.draw_networkx_edges(tree, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=16, node_size=760)
    borders = ['#7c3aed' if n == root else '#dc2626' if goal and n == goal else '#ffffff'
               for n in tree.nodes]
    nx.draw_networkx_nodes(tree, pos, ax=ax, node_color=node_colors,
                           edgecolors=borders, linewidths=2, node_size=760)
    labels = {n:f'{n}\nD{depths.get(n, 0)}' for n in tree.nodes}
    nx.draw_networkx_labels(tree, pos, labels=labels, ax=ax, font_size=9, font_weight='bold')
    ax.set_title(f'{result["method"]} search tree (D = depth)', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
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
if st.session_state.goal not in ['No goal']+nodes:
    st.session_state.goal = 'No goal'

a, b, c = st.columns(3)
start = a.selectbox('Starting node', nodes, key='start')
goal_choice = b.selectbox('Goal node (optional)', ['No goal']+nodes, key='goal')
method = c.selectbox('Algorithm', ['Both', 'BFS', 'DFS'], key='method')
goal = None if goal_choice == 'No goal' else goal_choice
results = {m:search(graph, start, goal, m) for m in (['BFS','DFS'] if method == 'Both' else [method])}


# Display each selected search algorithm.
for m, r in results.items():
    st.subheader(f'{m} search')
    st.write('**Expansion order:** ' + ' → '.join(r['order']))
    if goal:
        st.write('**Goal path:** ' + (' → '.join(r['path']) if r['found'] else 'Not reachable'))

    x, y, z = st.columns(3)
    x.metric('Expanded nodes', r['explored'])
    y.metric('Max frontier', r['max_frontier'])
    z.metric('Goal path edges', len(r['path'])-1 if r['found'] else '—')

    # Previous / Next buttons are independent for BFS and DFS.
    step_key = f'{m}_step'
    st.session_state.setdefault(step_key, 1)
    st.session_state[step_key] = max(1, min(st.session_state[step_key], len(r['trace'])))
    current_step = st.session_state[step_key]

    prev_col, info_col, next_col = st.columns([1, 1.25, 1])
    with prev_col:
        st.button('◀ Previous', key=f'{m}_prev', disabled=current
