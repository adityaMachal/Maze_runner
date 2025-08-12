#!/usr/bin/env python3
import time
import random
from collections import deque
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class CellType(Enum):
    WALL=0
    PATH=1
    START=2
    END=3
    VISITED=4
    CURRENT=5
    SOLUTION=6
    FRONTIER=7

class MazeGenerator:
    def __init__(self,width:int,height:int):
        self.width=width if width%2==1 else width+1
        self.height=height if height%2==1 else height+1
        self.maze=np.zeros((self.height,self.width),dtype=int)
    def generate_recursive_backtracking(self):
        self.maze.fill(CellType.WALL.value)
        start_r,start_c=1,1
        self.maze[start_r,start_c]=CellType.PATH.value
        stack=[(start_r,start_c)]
        dirs=[(0,2),(2,0),(0,-2),(-2,0)]
        while stack:
            cr,cc=stack[-1]
            neighbors=[]
            for dr,dc in dirs:
                nr,nc=cr+dr,cc+dc
                if 0<nr<self.height-1 and 0<nc<self.width-1 and self.maze[nr,nc]==CellType.WALL.value:
                    neighbors.append((nr,nc))
            if neighbors:
                nr,nc=random.choice(neighbors)
                wall_r,wall_c=cr+(nr-cr)//2,cc+(nc-cc)//2
                self.maze[wall_r,wall_c]=CellType.PATH.value
                self.maze[nr,nc]=CellType.PATH.value
                stack.append((nr,nc))
            else:
                stack.pop()
        self.maze[1,1]=CellType.START.value
        self.maze[self.height-2,self.width-2]=CellType.END.value
        return self.maze.copy()
    def generate_simple(self):
        grid=np.array([
            [0,0,0,0,0,0,0,0,0,0],
            [0,2,1,0,1,1,1,0,1,0],
            [0,0,1,0,1,0,1,0,1,0],
            [0,1,1,1,1,0,1,1,1,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,1,1,1,1,1,1,1,1,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,3,0],
            [0,0,0,0,0,0,0,0,0,0]
        ])
        self.maze=grid
        self.height,self.width=grid.shape
        return grid.copy()

class Pathfinder:
    def __init__(self,maze:np.ndarray):
        self.original_maze=maze.copy()
        self.maze=maze.copy()
        self.h,self.w=maze.shape
        self.steps=[]
        self.start,self.end=self._find_special_cells()
    def _find_special_cells(self):
        start=end=None
        for r in range(self.h):
            for c in range(self.w):
                if self.maze[r,c]==CellType.START.value:
                    start=(r,c)
                elif self.maze[r,c]==CellType.END.value:
                    end=(r,c)
        if not start or not end:
            raise ValueError("Maze must contain start (2) and end (3) cells")
        return start,end
    def _reset(self):
        self.maze=self.original_maze.copy()
        self.steps.clear()
    def _add_step(self,info):
        self.steps.append((self.maze.copy(),info))
    def _neighbors(self,r,c):
        for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<self.h and 0<=nc<self.w and self.maze[nr,nc]!=CellType.WALL.value:
                yield nr,nc
    def _reconstruct(self,came_from):
        path=[]; cur=self.end
        while cur!=self.start:
            path.append(cur)
            cur=came_from[cur]
        for r,c in path:
            if self.maze[r,c] not in (CellType.START.value,CellType.END.value):
                self.maze[r,c]=CellType.SOLUTION.value
    def dfs(self):
        self._reset()
        stack=[self.start]; visited=set(); came={}
        self._add_step("DFS start")
        while stack:
            r,c=stack.pop()
            if (r,c) in visited: continue
            visited.add((r,c))
            if (r,c)!=self.start: self.maze[r,c]=CellType.CURRENT.value
            self._add_step("DFS exploring")
            if (r,c)==self.end:
                self._reconstruct(came); self._add_step("DFS solved"); return True
            for nr,nc in self._neighbors(r,c):
                if (nr,nc) not in visited:
                    came[(nr,nc)]=(r,c)
                    stack.append((nr,nc))
                    if self.maze[nr,nc]==CellType.PATH.value:
                        self.maze[nr,nc]=CellType.FRONTIER.value
            if (r,c)!=self.start: self.maze[r,c]=CellType.VISITED.value
        self._add_step("DFS failed"); return False
    def bfs(self):
        self._reset()
        q=deque([self.start]); visited={self.start}; came={}
        self._add_step("BFS start")
        while q:
            r,c=q.popleft()
            if (r,c)!=self.start: self.maze[r,c]=CellType.CURRENT.value
            self._add_step("BFS exploring")
            if (r,c)==self.end:
                self._reconstruct(came); self._add_step("BFS solved"); return True
            for nr,nc in self._neighbors(r,c):
                if (nr,nc) not in visited:
                    visited.add((nr,nc)); came[(nr,nc)]=(r,c); q.append((nr,nc))
                    if self.maze[nr,nc]==CellType.PATH.value:
                        self.maze[nr,nc]=CellType.FRONTIER.value
            if (r,c)!=self.start: self.maze[r,c]=CellType.VISITED.value
        self._add_step("BFS failed"); return False

class Visualizer:
    color_map={
        CellType.WALL.value:'#000000',
        CellType.PATH.value:'#FFFFFF',
        CellType.START.value:'#00FF00',
        CellType.END.value:'#FF0000',
        CellType.VISITED.value:'#87CEEB',
        CellType.CURRENT.value:'#FFD700',
        CellType.SOLUTION.value:'#FF69B4',
        CellType.FRONTIER.value:'#98FB98'
    }
    def __init__(self,pathfinder:Pathfinder,algo_name:str,interval_ms:int):
        self.pf=pathfinder; self.algo_name=algo_name; self.interval=interval_ms
        self.fig,self.ax=plt.subplots(figsize=(8,6))
        self.cmap=ListedColormap([self.color_map[i] for i in range(len(self.color_map))])
        self.im=None
    def _init_plot(self):
        self.ax.set_title(f"{self.algo_name} - Maze Solver",fontsize=16)
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_aspect('equal')
        legend_elements=[mpatches.Patch(color=v,label=CellType(k).name.title()) for k,v in self.color_map.items()]
        self.ax.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1,0.5))
        self.im=self.ax.imshow(self.pf.steps[0][0],cmap=self.cmap,vmin=0,vmax=len(self.color_map)-1)
        return self.im,
    def _update(self,i):
        grid,info=self.pf.steps[i]
        self.im.set_array(grid)
        self.ax.set_title(f"{self.algo_name} - Step {i+1}/{len(self.pf.steps)}",fontsize=14)
        return self.im,
    def animate(self):
        ani=animation.FuncAnimation(self.fig,self._update,frames=len(self.pf.steps),init_func=self._init_plot,interval=self.interval,blit=True,repeat=False)
        plt.tight_layout(); plt.show()

class MazeSolverCLI:
    algorithms={'1':('DFS','Depth-First Search'),'2':('BFS','Breadth-First Search')}
    def banner(self):
        print("="*55); print("  🧩  MAZE SOLVER - Matplotlib Animation  🧩"); print("="*55)
    def choose(self,prompt,options):
        while True:
            choice=input(prompt).strip()
            if choice in options: return choice
            print(f"Enter one of {', '.join(options)}")
    def run(self):
        self.banner()
        while True:
            print("\nAlgorithms:")
            for k,(s,f) in self.algorithms.items(): print(f"  {k}. {f} ({s})")
            algo_choice=self.choose("Select algorithm (1-2): ",self.algorithms)
            algo_short,algo_full=self.algorithms[algo_choice]
            maze_type=self.choose("\nGenerate random maze? (y/n): ",['y','n'])
            if maze_type=='y':
                size_choice=self.choose("Size - small(1), medium(2), large(3): ",['1','2','3'])
                size_map={'1':15,'2':21,'3':31}; sz=size_map[size_choice]; mg=MazeGenerator(sz,sz); grid=mg.generate_recursive_backtracking()
            else:
                mg=MazeGenerator(10,9); grid=mg.generate_simple()
            pf=Pathfinder(grid)
            print("\nSolving...")
            start=time.time(); solved=getattr(pf,algo_short.lower())(); dur=time.time()-start
            print(f"Solved: {solved} in {dur:.3f}s, steps: {len(pf.steps)}")
            if not solved: print("No solution found.")
            speed_choice=self.choose("Animation speed - slow(1), normal(2), fast(3): ",['1','2','3'])
            speed_map={'1':400,'2':150,'3':30}; viz=Visualizer(pf,algo_full,speed_map[speed_choice]); viz.animate()
            again=self.choose("Run another? (y/n): ",['y','n']);
            if again=='n': break
        print("Bye!")

if __name__=='__main__':
    try:
        MazeSolverCLI().run()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
