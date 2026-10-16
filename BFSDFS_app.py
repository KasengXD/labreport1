import math
import time
from collections import deque
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title='BSD3513 | LAB REPORT 1',
    page_icon='🔎',
    layout='wide',
)


# ==========================================================
# DEFAULT GRAPH
# ==========================================================

GRAPH = {
    'A': ['B', 'D'],
    'B': ['C', 'E', 'G'],
    'C': ['A'],
    'D': ['C'],
    'E': ['H'],
    'F': [],
    'G': ['F'],
    'H': ['F', 'G'],
}


# ==========================================================
# NORMALIZE GRAPH
# ==========================================================

def normalize(graph):
    result = {
        str(k).strip().upper():
            sorted(set(str(v).strip().upper() for v in vs))
        for k, vs in graph.items()
    }

    for neighbors in list(result.values()):
        for node in neighbors:
            result.setdefault(node, [])

    return dict(sorted(result.items()))


# ==========================================================
# BFS AND DFS SEARCH
# ==========================================================

def search(graph, start, goal=None, method='BFS'):

    graph = normalize(graph)

    start = start.upper().strip()
    goal = goal.upper().strip() if goal else None

    if start not in graph or (goal and goal not in graph):
        raise ValueError('Start and goal must exist in the graph.')

    if method not in ('BFS', 'DFS'):
        raise ValueError('Method must be BFS or DFS.')

    # BFS uses a FIFO queue.
    # DFS uses a LIFO stack.
    frontier = deque([start]) if method == 'BFS' else [start]

    # BFS marks nodes when they enter the queue.
    # DFS marks nodes when they are expanded.
    discovered = {start} if method == 'BFS' else set()

    parent = {start: None}
    depth = {start: 0}

    order = []
    trace = []

    while frontier:

        # Remove a node from the frontier.
        node = frontier.popleft() if method == 'BFS' else frontier.pop()

        # Skip a duplicate entry in the DFS stack.
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

        # Mark the current node as expanded.
        order.append(node)

        # Do not expand the goal node.
        if node != goal:

            neighbors = (
                graph[node]
                if method == 'BFS'
                else list(reversed(graph[node]))
            )

            for neighbor in neighbors:

                if neighbor not in discovered:

                    if method == 'BFS':
                        discovered.add(neighbor)

                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1

                    frontier.append(neighbor)

        # Record the current search state.
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

    # Reconstruct the path from start to goal.
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
        'max_frontier': max(
            [1] + [len(t['frontier']) for t in trace]
        ),
    }


# ==========================================================
# CONVERT TEXT INTO A DIRECTED GRAPH
# ==========================================================

def edges_from_text(text):
    """
    One directed edge per line:
        A B
        A,B

    Isolated nodes:
        A
    """

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


# ==========================================================
# DEFAULT EDGE LIST
# ==========================================================

DEFAULT_EDGES = '\n'.join(
    f'{u} {v}' if v else u
    for u, neighbors in GRAPH.items()
    for v in (neighbors or [None])
)


# ==========================================================
# DEFAULT SETTINGS AND RESET BUTTON
# ==========================================================

DEFAULTS = {
    'edges': DEFAULT_EDGES,
    'start': 'A',
    'goal': 'No goal',
    'method': 'Both',
    'speed': 0.8,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_defaults():

    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    # Reset step sliders and autoplay button states.
    for key in [
        k for k in st.session_state
        if k.startswith(('BFS_', 'DFS_', 'play_'))
    ]:
        del st.session_state[key]


# ==========================================================
# BUILD NETWORKX GRAPH
# ==========================================================

def build_networkx_graph(graph):

    directed_graph = nx.DiGraph()

    for node, neighbors in graph.items():

        directed_graph.add_node(node)

        for neighbor in neighbors:
            directed_graph.add_edge(node, neighbor)

    return directed_graph


# ==========================================================
# IMPROVED GRAPH LAYOUT
# ==========================================================

def make_layout(directed_graph):
    """
    Create a stable layout for the entire animation.
    The default graph uses fixed positions.
    Custom graphs use a force-directed layout.
    """

    default_edges = {
        (u, v)
        for u, neighbors in GRAPH.items()
        for v in neighbors
    }

    if (
        set(directed_graph.nodes) == set(GRAPH)
        and set(directed_graph.edges) == default_edges
    ):

        # Fixed node positions for the default graph.
        # Nodes will not move between search steps.
        return {
            'A': (0.0, 0.0),
            'B': (1.35, 1.05),
            'C': (2.30, 0.05),
            'D': (0.65, -1.15),
            'E': (2.85, 2.05),
            'H': (4.15, 2.05),
            'G': (4.55, 0.55),
            'F': (5.90, 1.65),
        }

    # Automatically arrange a graph edited by the user.
    n_nodes = max(len(directed_graph), 1)
    spacing = 1.35 / math.sqrt(n_nodes)

    return nx.spring_layout(
        directed_graph,
        seed=13,
        k=spacing,
        iterations=180,
        weight=None,
    )


# ==========================================================
# IMPROVED SEARCH GRAPH VISUALIZATION
# ==========================================================

def draw_graph(result, step, directed_graph, pos, start, goal):

    item = result['trace'][step - 1]

    visited = set(item['visited'])
    frontier_order = item['frontier']

    # Nodes waiting in the queue or stack.
    frontier_nodes = set(frontier_order) - visited

    raw_current = item['current']
    is_skip = raw_current.endswith(' (skip)')
    current = raw_current.replace(' (skip)', '')

    # ------------------------------------------------------
    # 1. NODE COLOURS
    # ------------------------------------------------------

    colors = []
    border_colors = []
    border_widths = []

    for node in directed_graph.nodes:

        if node == current:
            colors.append('#10b981')       # Green: current node

        elif node in visited:
            colors.append('#93c5fd')       # Blue: expanded node

        elif node in frontier_nodes:
            colors.append('#fbbf24')        # Amber: frontier node

        else:
            colors.append('#e5e7eb')        # Grey: undiscovered

        # Starting and goal nodes have special borders.
        if goal and node == goal:
            border_colors.append('#dc2626')  # Red: goal
            border_widths.append(3.2)

        elif node == start:
            border_colors.append('#7c3aed')  # Purple: start
            border_widths.append(3.2)

        elif node == current:
            border_colors.append('#047857')
            border_widths.append(2.4)

        else:
            border_colors.append('#ffffff')
            border_widths.append(1.4)

    # ------------------------------------------------------
    # 2. HIGHLIGHT SEARCH-TREE EDGES
    # ------------------------------------------------------

    tree_edges = set()

    for child, parent in result['parents'].items():

        if (
            parent is not None
            and child in visited
            and parent in visited
        ):
            tree_edges.add((parent, child))

    # ------------------------------------------------------
    # 3. HIGHLIGHT THE GOAL PATH
    # ------------------------------------------------------

    path_edges = set()

    if goal and goal in visited and result['found']:

        path_edges = set(
            zip(result['path'], result['path'][1:])
        )

    # ------------------------------------------------------
    # 4. CREATE FIGURE
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.5, 6.4),
        dpi=130
    )

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    n_nodes = max(len(directed_graph), 1)

    node_size = max(
        250,
        min(1450, 16500 / n_nodes)
    )

    font_size = (
        10 if n_nodes <= 15
        else max(6, 10 - n_nodes // 18)
    )

    # ------------------------------------------------------
    # 5. DRAW DIRECTED EDGES
    # ------------------------------------------------------

    for source, target in directed_graph.edges:

        edge = (source, target)

        if edge in path_edges:

            edge_color = '#f59e0b'
            edge_width = 3.8
            edge_alpha = 1.0

        elif edge in tree_edges:

            edge_color = '#0f766e'
            edge_width = 2.4
            edge_alpha = 0.95

        else:

            edge_color = '#94a3b8'
            edge_width = 1.45
            edge_alpha = 0.78

        # Curve arrows in opposite directions so they
        # can be distinguished more easily.
        if directed_graph.has_edge(target, source):

            curvature = (
                0.17
                if str(source) < str(target)
                else -0.17
            )

        else:
            curvature = 0.045

        nx.draw_networkx_edges(
            directed_graph,
            pos,
            ax=ax,
            edgelist=[edge],
            arrows=True,
            arrowstyle='-|>',
            arrowsize=22 if n_nodes <= 20 else 14,
            edge_color=edge_color,
            width=edge_width,
            alpha=edge_alpha,
            connectionstyle=f'arc3,rad={curvature}',
            node_size=node_size,
        )

    # ------------------------------------------------------
    # 6. DRAW NODES
    # ------------------------------------------------------

    nx.draw_networkx_nodes(
        directed_graph,
        pos,
        ax=ax,
        nodelist=list(directed_graph.nodes),
        node_color=colors,
        node_size=node_size,
        edgecolors=border_colors,
        linewidths=border_widths,
    )

    # ------------------------------------------------------
    # 7. DRAW NODE LABELS
    # ------------------------------------------------------

    nx.draw_networkx_labels(
        directed_graph,
        pos,
        ax=ax,
        font_size=font_size,
        font_weight='bold',
        font_color='#0f172a',
    )

    # ------------------------------------------------------
    # 8. GRAPH TITLE
    # ------------------------------------------------------

    current_label = (
        f'{current} — duplicate skipped'
        if is_skip
        else current
    )

    ax.set_title(
        f"{result['method']} Search  |  "
        f"Step {step}/{len(result['trace'])}  |  "
        f"Current: {current_label}",
        fontsize=13,
        fontweight='bold',
        color='#0f172a',
        pad=18,
    )

    ax.axis('off')

    # ------------------------------------------------------
    # 9. GRAPH LEGEND
    # ------------------------------------------------------

    legend_items = [

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Current',
            markerfacecolor='#10b981',
            markeredgecolor='#047857',
            markersize=10,
        ),

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Expanded / visited',
            markerfacecolor='#93c5fd',
            markeredgecolor='#ffffff',
            markersize=10,
        ),

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Frontier',
            markerfacecolor='#fbbf24',
            markeredgecolor='#ffffff',
            markersize=10,
        ),

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Undiscovered',
            markerfacecolor='#e5e7eb',
            markeredgecolor='#ffffff',
            markersize=10,
        ),

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Start (purple ring)',
            markerfacecolor='#ffffff',
            markeredgecolor='#7c3aed',
            markeredgewidth=2,
            markersize=10,
        ),

        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label='Goal (red ring)',
            markerfacecolor='#ffffff',
            markeredgecolor='#dc2626',
            markeredgewidth=2,
            markersize=10,
        ),

        Line2D(
            [0], [0],
            color='#0f766e',
            linewidth=2.4,
            label='Search-tree edge',
        ),
    ]

    if goal:

        legend_items.append(
            Line2D(
                [0], [0],
                color='#f59e0b',
                linewidth=3.8,
                label='Goal path',
            )
        )

    ax.legend(
        handles=legend_items,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.025),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    return fig


# ==========================================================
# STREAMLIT PAGE
# ==========================================================

st.title(
    'BFS & DFS | Interactive Uninformed Search Algorithms'
)

st.caption(
    'BSD3513 Artificial Intelligence · Lab Report 1 · '
    'Wong Ka Seng · Breadth-First Search & Depth-First Search · '
    'Alphabetical tie-breaking · UMPSA'
)


# ==========================================================
# LEARNING GUIDE
# ==========================================================

with st.expander('Learning guide', expanded=False):

    st.markdown(
        '**BFS** explores level by level using a FIFO queue; '
        '**DFS** explores deeply using a LIFO stack. '
        'For adjacency-list implementations, both require '
        '**O(V + E)** time. BFS finds minimum-hop paths in '
        'unweighted graphs; DFS does not guarantee this.'
    )


# ==========================================================
# SIDEBAR: GRAPH EDITOR AND SETTINGS
# ==========================================================

with st.sidebar:

    st.header('Graph editor')

    edge_text = st.text_area(
        'Directed edges (one per line: A B)',
        key='edges',
        height=260,
    )

    st.caption(
        'A line containing only a node represents an '
        'isolated node. Edges are directed.'
    )

    st.slider(
        'Autoplay speed (seconds per step)',
        0.2,
        2.0,
        step=0.1,
        key='speed',
    )

    st.button(
        'Reset to default settings',
        on_click=reset_defaults,
        width='stretch',
    )


# ==========================================================
# VALIDATE GRAPH INPUT
# ==========================================================

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


# ==========================================================
# SEARCH CONTROLS
# ==========================================================

a, b, c = st.columns(3)

start = a.selectbox(
    'Starting node',
    nodes,
    key='start',
)

goal_choice = b.selectbox(
    'Goal node (optional)',
    ['No goal'] + nodes,
    key='goal',
)

method = c.selectbox(
    'Algorithm',
    ['Both', 'BFS', 'DFS'],
    key='method',
)

goal = None if goal_choice == 'No goal' else goal_choice


# ==========================================================
# RUN SELECTED SEARCH ALGORITHM(S)
# ==========================================================

results = {
    m: search(graph, start, goal, m)
    for m in (
        ['BFS', 'DFS']
        if method == 'Both'
        else [method]
    )
}


# Build the graph and layout once.
# Positions remain consistent throughout the animation.
directed_graph = build_networkx_graph(graph)
graph_pos = make_layout(directed_graph)


# ==========================================================
# DISPLAY SEARCH RESULTS
# ==========================================================

for method_name, result in results.items():

    st.subheader(f'{method_name} search')

    st.write(
        '**Expansion order:** '
        + ' → '.join(result['order'])
    )

    if goal:

        st.write(
            '**Goal path:** '
            + (
                ' → '.join(result['path'])
                if result['found']
                else 'Not reachable'
            )
        )

    # ------------------------------------------------------
    # SEARCH STATISTICS
    # ------------------------------------------------------

    x, y, z = st.columns(3)

    x.metric(
        'Expanded nodes',
        result['explored'],
    )

    y.metric(
        'Maximum recorded frontier',
        result['max_frontier'],
    )

    z.metric(
        'Goal path edges',
        (
            len(result['path']) - 1
            if result['found']
            else '—'
        ),
    )

    # ------------------------------------------------------
    # STEP SLIDER
    # ------------------------------------------------------

    step = st.slider(
        f'{method_name}: inspect expansion step',
        1,
        len(result['trace']),
        len(result['trace']),
        key=f'{method_name}_{len(result["trace"])}',
    )

    st.caption(
        'Use the step slider to inspect the search. '
        'Autoplay animates the graph step by step.'
    )

    play = st.button(
        f'▶ Autoplay {method_name}',
        key=f'play_{method_name}',
    )

    left, right = st.columns([1.25, 1])

    with left:

        graph_box = st.empty()
        frontier_box = st.empty()

    with right:

        st.write('**Step trace**')

        st.dataframe(
            pd.DataFrame(result['trace']),
            hide_index=True,
            width='stretch',
        )

    # ------------------------------------------------------
    # MANUAL INSPECTION / AUTOPLAY
    # ------------------------------------------------------

    steps_to_show = (
        range(1, len(result['trace']) + 1)
        if play
        else [step]
    )

    for current_step in steps_to_show:

        fig = draw_graph(
            result,
            current_step,
            directed_graph,
            graph_pos,
            start,
            goal,
        )

        graph_box.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

        trace_item = result['trace'][current_step - 1]

        frontier_text = (
            ' → '.join(trace_item['frontier'])
            if trace_item['frontier']
            else 'Empty'
        )

        current_text = trace_item['current']

        frontier_box.info(
            f"**Step {current_step}/{len(result['trace'])}**  ·  "
            f"Current operation: {current_text}  ·  "
            f"Depth: {trace_item['depth']}\n\n"
            f"**Frontier "
            f"({'FIFO queue' if method_name == 'BFS' else 'LIFO stack'}):** "
            f"{frontier_text}"
        )

        if play:
            time.sleep(st.session_state.speed)


# ==========================================================
# BFS VS DFS COMPARISON TABLE
# ==========================================================

st.divider()

if method == 'Both':

    st.subheader('BFS vs DFS comparison')

    bfs = results['BFS']
    dfs = results['DFS']

    rows = {

        'Data structure': (
            'FIFO queue',
            'LIFO stack',
        ),

        'Expansion order': (
            ' → '.join(bfs['order']),
            ' → '.join(dfs['order']),
        ),

        'Expanded nodes': (
            bfs['explored'],
            dfs['explored'],
        ),

        'Maximum recorded frontier': (
            bfs['max_frontier'],
            dfs['max_frontier'],
        ),

        'Deepest level reached': (
            max(bfs['depths'].values()),
            max(dfs['depths'].values()),
        ),
    }

    if goal:

        rows['Goal path'] = (
            ' → '.join(bfs['path']) or 'Not reachable',
            ' → '.join(dfs['path']) or 'Not reachable',
        )

        rows['Goal path edges'] = (
            len(bfs['path']) - 1 if bfs['found'] else '—',
            len(dfs['path']) - 1 if dfs['found'] else '—',
        )

    comparison = pd.DataFrame(
        [
            (key, str(value[0]), str(value[1]))
            for key, value in rows.items()
        ],
        columns=['Measure', 'BFS', 'DFS'],
    )

    st.dataframe(
        comparison,
        hide_index=True,
        width='stretch',
    )

    if (
        goal
        and bfs['found']
        and dfs['found']
        and len(bfs['path']) < len(dfs['path'])
    ):

        st.success(
            f'BFS found a shorter path to {goal} '
            f'({len(bfs["path"]) - 1} edges vs '
            f'{len(dfs["path"]) - 1}), '
            'as expected for unweighted graphs.'
        )
