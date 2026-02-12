import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Shuffle, AlertTriangle } from 'lucide-react';
import './App.css';

// --- Constants & Utilities ---
const CANVAS_SIZE = 600;
const NODE_RADIUS = 20;
const MAX_ITERATIONS = 1000; // Safety brake for infinite loops

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const getGroundTruth = (nodes, edges, startId, endId) => {
  // 1. Min Edges (BFS style)
  const bfs = new GraphSearch(nodes, edges, startId, endId, 'BFS', { checkDuplicates: true });
  const minEdges = bfs.history.find(h => h.status === 'found')?.parents;
  
  // 2. Min Cost (Dijkstra style)
  const dijkstra = new GraphSearch(nodes, edges, startId, endId, 'Dijkstra', { checkDuplicates: true });
  const minCostSnap = dijkstra.history.find(h => h.status === 'found');
  
  // Helper to reconstruct path length/cost
  const getMetrics = (parents, isWeighted) => {
    if (!parents) return Infinity;
    let curr = endId;
    let count = 0;
    let cost = 0;
    while (curr !== startId) {
      let p = parents[curr];
      if (p === undefined) return Infinity;
      if (isWeighted) {
        const edge = edges.find(e => (e.source === p && e.target === curr) || (e.target === p && e.source === curr));
        cost += edge.weight;
      }
      count++;
      curr = p;
    }
    return isWeighted ? cost : count;
  };

  return {
    trueMinEdges: getMetrics(minEdges, false),
    trueMinCost: getMetrics(minCostSnap?.parents, true)
  };
};

// --- Graph Generation ---
const generateRandomGraph = (numNodes = 15) => {
  const nodes = [];
  const edges = [];
  
  // 1. Create a shuffled pool of alphabet letters
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const shuffledAlphabet = alphabet.sort(() => Math.random() - 0.5);
  
  // 2. Generate Nodes with randomized labels
  for (let i = 0; i < numNodes; i++) {
    let x, y, tooClose;
    do {
      x = Math.random() * (CANVAS_SIZE - 100) + 50;
      y = Math.random() * (CANVAS_SIZE - 100) + 50;
      tooClose = nodes.some(n => distance(n, { x, y }) < 60);
    } while (tooClose);
    
    nodes.push({ 
      id: i, 
      x, 
      y, 
      label: shuffledAlphabet[i] // Assign a random unique letter
    });
  }

  // 3. Connect Nodes (Nearest neighbor strategy)
  nodes.forEach((node, i) => {
    const neighbors = nodes
      .map((n, idx) => ({ idx, dist: distance(node, n) }))
      .filter(n => n.idx !== i)
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 3);

    neighbors.forEach(n => {
      const edgeExists = edges.some(e => 
        (e.source === i && e.target === n.idx) || 
        (e.source === n.idx && e.target === i)
      );
      if (!edgeExists) {
        edges.push({ source: i, target: n.idx, weight: Math.floor(n.dist) });
      }
    });
  });

  // 4. Randomly pick start and end indices
  const startIdx = Math.floor(Math.random() * numNodes);
  let endIdx;
  do {
    endIdx = Math.floor(Math.random() * numNodes);
  } while (endIdx === startIdx);

  return { nodes, edges, start: startIdx, end: endIdx };
};

// --- Preset Scenarios for A* ---
const PRESETS = {
  nonAdmissible: {
    nodes: [
      { id: 0, x: 50,  y: 300, label: 'S', h: 5 },   // Start
      { id: 1, x: 200, y: 150, label: 'A', h: 1000 }, // Overestimates!
      { id: 2, x: 200, y: 450, label: 'B', h: 0 },   // Looks like the goal is right here
      { id: 3, x: 400, y: 300, label: 'C', h: 0 },
      { id: 4, x: 550, y: 300, label: 'G', h: 0 }    // Goal
    ],
    edges: [
      { source: 0, target: 1, weight: 1 },   // S -> A
      { source: 0, target: 2, weight: 2 },   // S -> B
      { source: 1, target: 3, weight: 2 },   // A -> C
      { source: 2, target: 3, weight: 100 }, // B -> C (The "Expensive" path)
      { source: 3, target: 4, weight: 1 }    // C -> G
    ],
    start: 0, end: 4, 
    description: "Heuristic at A (100) is > actual cost to goal (3). A* will abandon the optimal path through A and take the expensive path through B."
  },
  inconsistent: {
    nodes: [
      { id: 0, x: 50,  y: 300, label: 'S', h: 10 },
      { id: 1, x: 200, y: 150, label: 'A', h: 100 }, // Inconsistent: h(A) > dist(A,C) + h(C)
      { id: 2, x: 200, y: 450, label: 'B', h: 0 },
      { id: 3, x: 400, y: 300, label: 'C', h: 0 },
      { id: 4, x: 550, y: 300, label: 'G', h: 0 }
    ],
    edges: [
      { source: 0, target: 1, weight: 1 },   // S -> A
      { source: 0, target: 2, weight: 1 },   // S -> B
      { source: 1, target: 3, weight: 1 },   // A -> C
      { source: 2, target: 3, weight: 10 },  // B -> C
      { source: 3, target: 4, weight: 100 }  // C -> G
    ],
    start: 0, end: 4,
    description: "When Duplicate Searching is ON, A* fails to find the optimal path because it 'locks' C via the sub-optimal path through B first."
  }
};

// --- Algorithms ---

class GraphSearch {
  constructor(nodes, edges, startId, endId, type, options = {}) {
    this.nodes = nodes;
    this.edges = edges;
    this.startId = startId;
    this.endId = endId;
    this.type = type; // 'BFS', 'DFS', 'Dijkstra', 'AStar', 'BiBFS'
    this.options = options; // { checkDuplicates: bool, heuristicType: 'euclidean' | 'preset' }
    this.adj = this.buildAdjacency();
    
    this.history = []; // Array of snapshots
    this.run();
  }

  buildAdjacency() {
    const adj = {};
    this.nodes.forEach(n => adj[n.id] = []);
    this.edges.forEach(e => {
      adj[e.source].push({ to: e.target, weight: e.weight });
      adj[e.target].push({ to: e.source, weight: e.weight });
    });
    return adj;
  }

  getHeuristic(nodeId) {
    const node = this.nodes.find(n => n.id === nodeId);
    const goal = this.nodes.find(n => n.id === this.endId);
    
    if (this.options.heuristicType === 'preset' && node.h !== undefined) return node.h;
    if (this.options.heuristicType === 'zero') return 0; // Dijkstra is A* with h=0
    
    // Default: Euclidean (Admissible & Consistent)
    return Math.floor(distance(node, goal));
  }

  snapshot(queue, visited, parents, current, status = "exploring") {
    // Deep copy queue for visualization
    const queueCopy = queue.map(item => ({...item})); 
    
    this.history.push({
      queue: queueCopy,
      visited: new Set(visited),
      parents: {...parents},
      current,
      status // 'exploring', 'found', 'failed'
    });
  }

  run() {
    if (this.type === 'BiBFS') {
      this.runBiBFS();
      return;
    }
  
    let queue = [{ id: this.startId, cost: 0, priority: 0, pathLength: 0 }];
    let visited = new Set();
    let parents = {};
    let iterations = 0;
  
    if (this.startId === this.endId) {
      this.snapshot(queue, visited, parents, this.startId, 'found');
      return;
    }
  
    while (queue.length > 0) {
      iterations++;
      
      if (iterations > MAX_ITERATIONS) {
        this.snapshot(queue, visited, parents, null, 'limit_reached');
        return;
      }
  
      // Sort for Priority Queues (A* / Dijkstra)
      if (['Dijkstra', 'AStar'].includes(this.type)) {
        queue.sort((a, b) => a.priority - b.priority);
      }
  
      // Identify the head for the snapshot (Last for DFS, First for others)
      let head = this.type === 'DFS' ? queue[queue.length - 1] : queue[0];
      this.snapshot(queue, visited, parents, head.id, 'exploring');
  
      // --- THE CORE FIX ---
      let current = this.type === 'DFS' ? queue.pop() : queue.shift();
  
      if (this.options.checkDuplicates && visited.has(current.id)) {
        continue;
      }
      
      visited.add(current.id);
  
      // Dijkstra and A* Goal Check (At Pop)
      if (['Dijkstra', 'AStar'].includes(this.type) && current.id === this.endId) {
        this.snapshot(queue, visited, parents, current.id, 'found');
        return;
      }
  
      // Sort neighbors Alphabetically
      const neighbors = (this.adj[current.id] || []).sort((a, b) => {
        const labelA = this.nodes.find(n => n.id === a.to).label;
        const labelB = this.nodes.find(n => n.id === b.to).label;
        return labelA.localeCompare(labelB);
      });
  
      // Reverse for DFS so the 'lowest' letter is at the END of the array (Top of Stack)
      if (this.type === 'DFS') {
        neighbors.reverse();
      }
  
      for (let edge of neighbors) {
        const neighborId = edge.to;
        const newCost = current.cost + edge.weight;
        const newPathLength = (current.pathLength || 0) + 1;
  
        const inFrontier = queue.some(item => item.id === neighborId);
        const shouldSkip = this.options.checkDuplicates && (visited.has(neighborId) || inFrontier);
  
        if (!shouldSkip) {
          parents[neighborId] = current.id;
  
          if (!['Dijkstra', 'AStar'].includes(this.type) && neighborId === this.endId) {
            queue.push({ id: neighborId, cost: newCost, priority: 0, pathLength: newPathLength });
            this.snapshot(queue, visited, parents, neighborId, 'found');
            return;
          }
  
          const hValue = this.type === 'AStar' ? this.getHeuristic(neighborId) : 0;
          queue.push({ 
            id: neighborId, 
            cost: newCost, 
            priority: newCost + hValue, 
            h: hValue, 
            pathLength: newPathLength 
          });
        }
      }
    }
    this.snapshot(queue, visited, parents, null, 'failed');
  }
  runBiBFS() {
    let qStart = [this.startId];
    let qEnd = [this.endId];
    let visitedStart = new Set([this.startId]);
    let visitedEnd = new Set([this.endId]);
    let parentStart = {};
    let parentEnd = {};
    
    let iterations = 0;

    while (qStart.length > 0 && qEnd.length > 0) {
      iterations++;
      
      // Expand Start Side
      const currStart = qStart.shift();
      this.history.push({ 
        queue: [...qStart.map(id=>({id, side:'start'})), ...qEnd.map(id=>({id, side:'end'}))], 
        visited: new Set([...visitedStart, ...visitedEnd]), 
        parents: {...parentStart, ...parentEnd},
        current: currStart,
        status: 'exploring',
        activeSide: 'start'
      });

      if (visitedEnd.has(currStart)) {
        // Intersection found!
        // We need to merge parents to show path
        this.history[this.history.length-1].status = 'found';
        this.history[this.history.length-1].intersect = currStart;
        return;
      }

      (this.adj[currStart] || []).forEach(edge => {
        if (!visitedStart.has(edge.to)) {
          visitedStart.add(edge.to);
          parentStart[edge.to] = currStart;
          qStart.push(edge.to);
        }
      });

      // Expand End Side
      const currEnd = qEnd.shift();
      this.history.push({ 
        queue: [...qStart.map(id=>({id, side:'start'})), ...qEnd.map(id=>({id, side:'end'}))], 
        visited: new Set([...visitedStart, ...visitedEnd]), 
        parents: {...parentStart, ...parentEnd},
        current: currEnd,
        status: 'exploring',
        activeSide: 'end'
      });

      if (visitedStart.has(currEnd)) {
        // Intersection found
        this.history[this.history.length-1].status = 'found';
        this.history[this.history.length-1].intersect = currEnd;
        return;
      }

      (this.adj[currEnd] || []).forEach(edge => {
        if (!visitedEnd.has(edge.to)) {
          visitedEnd.add(edge.to);
          parentEnd[edge.to] = currEnd;
          qEnd.push(edge.to);
        }
      });
    }
  }
}

// --- React Component ---

const App = () => {
  // State
  const [graph, setGraph] = useState(generateRandomGraph());
  const [algoType, setAlgoType] = useState('BFS');
  const [checkDuplicates, setCheckDuplicates] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [stepIndex, setStepIndex] = useState(0);
  const [history, setHistory] = useState([]);
  const [presetName, setPresetName] = useState('random');
  // Inside your App component, calculate ground truth whenever the graph changes
  const [groundTruth, setGroundTruth] = useState({ trueMinEdges: 0, trueMinCost: 0 });

  useEffect(() => {
    const truth = getGroundTruth(graph.nodes, graph.edges, graph.start, graph.end);
    setGroundTruth(truth);
  }, [graph]);

  const timerRef = useRef(null);

  // Re-run algorithm when settings change
  useEffect(() => {
    runAlgorithm();
  }, [graph, algoType, checkDuplicates]);

  const runAlgorithm = () => {
    setIsPlaying(false);
    setStepIndex(0);
    
    let options = { 
      checkDuplicates, 
      heuristicType: presetName === 'random' ? 'euclidean' : 'preset'
    };

    if (algoType === 'Dijkstra') options.heuristicType = 'zero';

    const search = new GraphSearch(graph.nodes, graph.edges, graph.start, graph.end, algoType, options);
    setHistory(search.history);
  };

  // Playback Control
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setStepIndex(prev => {
          if (prev >= history.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000 - speed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, history.length, speed]);

  const handleShuffle = () => {
    setPresetName('random');
    setGraph(generateRandomGraph());
  };

  const loadPreset = (name) => {
    setPresetName(name);
    setGraph(PRESETS[name]);
  };

  // Get current state
  const currentStep = history[stepIndex] || { queue: [], visited: new Set(), parents: {}, status: 'start' };
  // Helper to reconstruct the path for the CURRENT node being explored
  const getCurrentTrace = () => {
    if (!currentStep.current) return [];
    const trace = [];
    let curr = currentStep.current;
    while (curr !== undefined && curr !== graph.start) {
      trace.push(curr);
      curr = currentStep.parents[curr];
      if (trace.length > graph.nodes.length) break; // Safety against cycles
    }
    trace.push(graph.start);
    return trace;
  };
  
  const currentTrace = getCurrentTrace();
  
  // Reconstruct Path
  const getPath = () => {
    if (currentStep.status !== 'found') return [];
    
    const path = [];
    
    if (algoType === 'BiBFS' && currentStep.intersect !== undefined) {
      // Reconstruct BiBFS path
      // From Start -> Intersect
      let curr = currentStep.intersect;
      while (curr !== graph.start) {
        path.unshift(curr);
        curr = currentStep.parents[curr];
        if(!curr) break; // safety
      }
      path.unshift(graph.start);
      
      // From Intersect -> End (Logic depends on parent map structure, simplified here)
      // Note: In a real BiBFS visualization we need distinct parent maps for start/end
      // to avoid overwrites. My simplified BiBFS merges them, which might look odd 
      // if paths overlap, but for visualization it works if we just highlight nodes.
      return [...path]; 
    }

    // Standard Path
    let curr = graph.end;
    while (curr !== undefined && curr !== graph.start) {
      path.push(curr);
      curr = currentStep.parents[curr];
    }
    path.push(graph.start);
    return path;
  };
  
  const path = getPath();

  // Stats
  const maxQueueSize = history.reduce((max, step) => Math.max(max, step.queue.length), 0);
  const nodesExplored = currentStep.visited.size;


  // Inside the Sidebar render:
  const currentPath = getPath();
  const currentEdges = currentPath.length > 0 ? currentPath.length - 1 : Infinity;

  // Calculate current cost
  let currentCost = 0;
  for (let i = 0; i < currentPath.length - 1; i++) {
    const u = currentPath[i];
    const v = currentPath[i+1];
    const edge = graph.edges.find(e => (e.source === u && e.target === v) || (e.target === u && e.source === v));
    currentCost += edge?.weight || 0;
  }
  
  return (
    <div className="app-container">
      <header className="header">
        <h1>Graph Search Visualizer</h1>
        <div className="controls-top">
          <select value={algoType} onChange={e => setAlgoType(e.target.value)}>
            <option value="BFS">BFS</option>
            <option value="DFS">DFS</option>
            <option value="Dijkstra">Dijkstra</option>
            <option value="AStar">A* Search</option>
          </select>

          <label className="checkbox-wrapper">
            <input 
              type="checkbox" 
              checked={checkDuplicates} 
              onChange={e => setCheckDuplicates(e.target.checked)} 
            />
            Prevent Re-visits (Visited Set)
          </label>

          <button onClick={handleShuffle} className="btn-icon"><Shuffle size={16}/> Random Graph</button>
        </div>
        
        {algoType === 'AStar' && (
          <div className="preset-bar">
            <span>A* Presets: </span>
            <button className={presetName==='random'?'active':''} onClick={handleShuffle}>Consistent (Euclidean)</button>
            <button className={presetName==='inconsistent'?'active':''} onClick={() => loadPreset('inconsistent')}>Inconsistent (Trap)</button>
            <button className={presetName==='nonAdmissible'?'active':''} onClick={() => loadPreset('nonAdmissible')}>Non-Admissible</button>
          </div>
        )}
      </header>

      <div className="main-content">
        <div className="canvas-wrapper">
          <svg 
            viewBox={`0 0 ${CANVAS_SIZE} ${CANVAS_SIZE}`} 
            preserveAspectRatio="xMidYMid meet"
            className="graph-svg"
          >
            {/* Edges */}
            {graph.edges.map((e, i) => {
              const u = graph.nodes.find(n => n.id === e.source);
              const v = graph.nodes.find(n => n.id === e.target);
              
              // 1. Final Path (Gold/Orange)
              const isFinalPath = path.includes(e.source) && path.includes(e.target) && 
                                  (currentStep.parents[e.source] === e.target || currentStep.parents[e.target] === e.source);
              
              // 2. Current "Ghost" Trace (Thin Blue/White)
              const isTraceEdge = currentTrace.includes(e.source) && currentTrace.includes(e.target) &&
                                  (currentStep.parents[e.source] === e.target || currentStep.parents[e.target] === e.source);

              return (
                <g key={i}>
                  {/* Base Edge */}
                  <line x1={u.x} y1={u.y} x2={v.x} y2={v.y} className="edge" />
                  
                  {/* Ghost Trace (Thin Line) */}
                  {isTraceEdge && !isFinalPath && (
                    <line x1={u.x} y1={u.y} x2={v.x} y2={v.y} className="edge-trace" />
                  )}
                  
                  {/* Final Path (Bold Line) */}
                  {isFinalPath && currentStep.status === 'found' && (
                    <line x1={u.x} y1={u.y} x2={v.x} y2={v.y} className="edge-path" />
                  )}
                  
                  <text x={(u.x + v.x)/2} y={(u.y + v.y)/2} className="edge-label" dy={-5}>{e.weight}</text>
                </g>
              );
            })}
            {/* Nodes */}
            {graph.nodes.map(n => {
              const isStart = n.id === graph.start;
              const isEnd = n.id === graph.end;
              const isCurrent = currentStep.current === n.id;
              const isVisited = currentStep.visited.has(n.id);
              const isFrontier = currentStep.queue.some(item => item.id === n.id);
              
              let classes = "node";
              if (isStart) classes += " start";
              else if (isEnd) classes += " end";
              else if (isCurrent) classes += " current";
              else if (isFrontier) classes += " frontier"; // Frontier should take priority over visited if both exist
              else if (isVisited) classes += " visited";
              return (
                <g key={n.id} transform={`translate(${n.x},${n.y})`}>
                  <circle r={NODE_RADIUS} className={classes} />
                  <text dy={5} className="node-text">{n.label}</text>
                  {algoType === 'AStar' && (
                    <text dy={-25} className="heuristic-text">h: {algoType === 'AStar' ? (n.h ?? Math.floor(distance(n, graph.nodes.find(no=>no.id===graph.end)))) : ''}</text>
                  )}
                </g>
              );
            })}
          </svg>
          
          <div className="playback-controls">
            <button onClick={() => setStepIndex(0)}><RotateCcw size={20}/></button>
            <button onClick={() => setStepIndex(Math.max(0, stepIndex-1))}><ChevronLeft size={20}/></button>
            <button onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? <Pause size={20}/> : <Play size={20}/>}
            </button>
            <button onClick={() => setStepIndex(Math.min(history.length-1, stepIndex+1))}><ChevronRight size={20}/></button>
            
            <input 
              type="range" 
              min="0" max={history.length - 1 || 0} 
              value={stepIndex} 
              onChange={e => setStepIndex(parseInt(e.target.value))}
              className="timeline-slider"
            />
          </div>
        </div>

        <div className="sidebar">
          {/* 1. Status & Detailed Statistics Panel */}
          <div className="panel status-panel">
            <h3>Status & Statistics</h3>
            <div className={`status-badge ${currentStep.status}`}>
              {currentStep.status.toUpperCase().replace('_', ' ')}
            </div>
            
            {currentStep.status === 'limit_reached' && (
              <p className="warning"><AlertTriangle size={14}/> Cycle Detected / Max Iterations</p>
            )}

            <div className="stats-list">
              <div className="stat-row">
                <label>Current Node</label>
                <span>{graph.nodes.find(n => n.id === currentStep.current)?.label || 'None'}</span>
              </div>
              <div className="stat-row">
                <label>Partial Path</label>
                <span className="path-text">
                  {getPath().map(id => graph.nodes.find(n => n.id === id).label).reverse().join('→') || '-'}
                </span>
              </div>
              
              <hr className="sidebar-divider" />
              
              <div className="stat-row">
                <label>Iterations (steps)</label>
                <span>{stepIndex} / {history.length - 1}</span>
              </div>
              <div className="stat-row">
                <label>Explored (pops)</label>
                <span>{currentStep.visited?.size || 0}</span>
              </div>
              <div className="stat-row">
                <label>Frontier Avg Size</label>
                <span>{(history.slice(0, stepIndex + 1).reduce((acc, h) => acc + h.queue.length, 0) / (stepIndex + 1)).toFixed(2)}</span>
              </div>
              <div className="stat-row">
                <label>Frontier Max Size</label>
                <span>{Math.max(...history.slice(0, stepIndex + 1).map(h => h.queue.length), 0)}</span>
              </div>
              
              <hr className="sidebar-divider" />
              
              <div className="stat-row">
                <label>Found Goal?</label>
                <span>{currentStep.status === 'found' ? '✅ Yes' : '❌ No'}</span>
              </div>
              <div className="stat-row">
                <label>Path Length (edges)</label>
                <span>{getPath().length > 0 ? getPath().length - 1 : 0}</span>
              </div>
              
              {/* Optimality Logic */}
              <div className="stat-row">
                <label>Shortest Path (Edges)?</label>
                <span>
                  {currentStep.status === 'found' 
                    ? (currentEdges <= groundTruth.trueMinEdges ? '✅ Yes' : '❌ No') 
                    : 'N/A'}
                </span>
              </div>

              <div className="stat-row">
                <label>Least Cost (Weights)?</label>
                <span>
                  {currentStep.status === 'found' 
                    ? (currentCost <= groundTruth.trueMinCost ? '✅ Yes' : '❌ No') 
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* 2. Frontier (Queue/Stack) Panel */}
          <div className="panel queue-panel">
            <h3>{algoType === 'DFS' ? 'Stack (LIFO)' : 'Queue (FIFO/Priority)'}</h3>
            <div className="queue-list">
              {currentStep.queue.map((item, i) => (
                <div key={i} className="queue-item">
                  <strong>{graph.nodes.find(n => n.id === item.id)?.label}</strong>
                  <span className="priority">
                    {['BFS', 'DFS', 'BiBFS'].includes(algoType) 
                      ? `Len: ${item.pathLength || 0}` 
                      : (
                        <span>
                          f: {item.priority?.toFixed(0)} 
                          {/* This shows the breakdown: g + h */}
                          <small style={{ opacity: 0.7, marginLeft: '4px' }}>
                            ({(item.priority - (item.h || 0)).toFixed(0)} + {item.h || 0})
                          </small>
                        </span>
                      )
                    }
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* 3. Legend Panel */}
          <div className="panel legend">
            <h3>Legend</h3>
            <div className="legend-item"><span className="dot start"></span> Start Node</div>
            <div className="legend-item"><span className="dot end"></span> Goal Node</div>
            <div className="legend-item"><span className="dot current"></span> Current Head</div>
            <div className="legend-item"><span className="dot visited"></span> Visited Set</div>
            <div className="legend-item"><span className="dot frontier"></span> Frontier (In Queue)</div>
            <div className="legend-item">
            <span className="dot" style={{ 
              height: '2px', 
              width: '14px', 
              backgroundColor: 'var(--accent)', 
              borderRadius: '0', 
              border: 'none' 
            }}></span> 
            Current Trace Path
          </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;