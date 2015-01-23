 /* Program to implement Dijkstra's shortest path algorith
 * using both array and fibonacci heap
 * Providing two operational modes, Random and User mode
 * Author Mayank Kumar Dadheech
 * UFID: 6980-5273
 * Advanced Data structure Spring-2014 Project
 */
import java.util.*;
import java.io.*;

public class dijikstra 
{

	public static void main(String args[]) throws IOException
    {
        UndirectedGraph<Integer> grph = new UndirectedGraph<Integer>();
        Graph theGraph = new Graph();
	
		if(args[0].equals("-r"))						//Check for Random mode
		{
			/*###  RANDOM MODE  ###*/
		
			int[][] weight = new int [6000][6000];  		//Array for storing the weights of the connected graph
		   
			int numNodes = Integer.parseInt(args[1]); 	  	//Storing number of nodes
			double den = Double.parseDouble(args[2]);		//Storing density of graph
			int sNode = Integer.parseInt(args[3]);			//Setting Source node number
			int maxden=(numNodes*(numNodes-1))/2;   		//Number of edges when density = 100%
			int m=(int)((maxden*den)/100);                 	//Number of edges when density = den%
			final WeightedGraph grph_simple = new WeightedGraph (numNodes);

			for(int i=0; i<numNodes; i++)					//Populating graph with nodes/vertices
			{
				grph.addNode(i);
				String rt = String.valueOf(i);
				theGraph.addVertex(rt);
				grph_simple.setLabel(i,i);
			}

			
			int cc = 0;
			while(cc<numNodes) 								//Generate Random Edges and Check Graph Connectivity
			{
				for(int c= 0;c<m;c++)
				{
					int aa=randomnumber(numNodes);			//Random Node 1
					int bb=randomnumber(numNodes);			//Random Node 2
					int rr=randomnumber(1000)+1;			//Random Cost generation for an Edge
					grph.addEdge(aa, bb, rr);				//Adding Edge in the Graph
					theGraph.addEdge(aa,bb);
					grph_simple.addEdge(aa,bb,rr);
					weight[aa][bb]=rr;     					//Setting cost matrix, both ways
					weight[bb][aa]=rr;  				
				}
				
				cc=theGraph.dfs();  						//Applying DFS to check whether the generated graph is CONNECTED
			}
			
			/*IMPLEMENTING DIJIKSTRA'S ALGORITHM USING SIMPLE MODE IN RANDOM MODE*/
			Map<Integer, Double> result = new HashMap<Integer,Double>();
			long simp_start_time, simp_stop_time = 0;
			simp_start_time = System.currentTimeMillis();							//Time assignment for performance check
			result = Dijkstra.shortestPaths(grph, sNode);       				 	//Implement DIJKSTRA'S ALGORITHM using Simple way
			simp_stop_time = System.currentTimeMillis();
			long simp_run_time =  simp_stop_time- simp_start_time;
			System.out.println("Simple scheme in random mode takes "+simp_run_time+" Time ");
		/*	System.out.println("Result for Dijkstra's Algorithm using Simple scheme in random mode :");
		
			Set<Map.Entry<Integer,Double>> set = result.entrySet();		//_mkd_trying to iterate through graph map
			
			for(Map.Entry<Integer,Double> me : set)
			{
				Integer q = me.getKey();
				Double e = me.getValue();
				if(e != null)
				{
					System.out.println(q + "'s weight from source  " + e);
				}
        	}
		 */
		 
			/*IMPLEMENTING DIJIKSTRA'S ALGORITHM USING FIBONACCI SCHEME IN RANDOM MODE*/
			long fib_start_time, fib_stop_time = 0;
			fib_start_time = System.currentTimeMillis();
			int [] pred = Dijkstra_Two.dijkstra (grph_simple, sNode);
			fib_stop_time = System.currentTimeMillis();
			long fib_run_time =  fib_stop_time - fib_start_time;
			System.out.println("Fibonacci scheme in random mode takes "+fib_run_time+" Time ");
			
			//DijkstraFib.printPath (grph_simple, pred, sNode, n);
					
			
			
		}
	
		
//#####################################################################################################
		
		//Check for Fibonacci Heap in User mode     
	    else if(args[0].equals("-f"))
		{
			File file = new File(args[1]);  			//input file
			BufferedReader bufRdr = new BufferedReader(new FileReader(file));  
			//read from the text file
			String line = null;     					//initializing line
			int row = 0;        						//initialing row
			int col = 0;        						//initializing column
			int sNode=0;
			int numVert=0;
			int numEdge=0;
			
			if((line = bufRdr.readLine()) != null)
			{
				StringTokenizer st = new StringTokenizer(line," ");
				sNode = Integer.parseInt(st.nextToken());
				if((line = bufRdr.readLine()) != null)
				{
					StringTokenizer st1 = new StringTokenizer(line," ");
					numVert = Integer.parseInt(st1.nextToken());
					numEdge = Integer.parseInt(st1.nextToken());
				}
				else
				{
					System.out.print("looks like your file doesn't have second row, number of vertices and edges? ");
				}
			}
			
			else
			{
				System.out.println("looks like the file doesn't have valid value for source node");
			}
			
			int maxVert=((numVert *(numVert-1)) /2);
			double [][] arr = new double[maxVert][3];
			String [][] num1 = new String [maxVert][3];
			
			while((line = bufRdr.readLine()) != null && row<=(numEdge+2) )
			{ 
				StringTokenizer st = new StringTokenizer(line," ");
				while (st.hasMoreTokens())
				{
					num1[row][col] = st.nextToken(); 						//get next token and store it in the array
					arr[row][col]=Double.parseDouble(num1[row][col]);  		//store the data read from the file in an 2-dimentional array
																			//type casting the strings stored from the file to double
					col++;      											//increment the column counter
					
				}
				col = 0;    												//the first column of the next row
				row++;      												//next row
			}     
   
			bufRdr.close();
			
			for(int q=0;q<numVert;q++)					//populate graph by adding vertices
			{				
				grph.addNode(q);
				theGraph.addVertex(String.valueOf(q));
			}

			for(int w=0;w<numEdge;w++)
			{
				/*add edges between vertices of the graph with the data from the file*/
				grph.addEdge(Integer.parseInt(num1[w][0]), Integer.parseInt(num1[w][1]), Double.parseDouble(num1[w][2]));
				theGraph.addEdge(Integer.parseInt(num1[w][0]),Integer.parseInt(num1[w][1]));
			}

			/*IMPLEMENTING DIJIKSTRA'S ALGORITHM USING FIBONACCI HEAP IN USER MODE*/
			//call Fibonacci and print result
			Map<Integer, Double> result = new HashMap<Integer,Double>();
			result = Dijkstra.shortestPaths(grph, sNode);       				 //Calling DIJKSTRA'S ALGORITHM using FIBONACCI HEAP
			System.out.println("Result for Dijkstra's Algorithm using Fibonacci Heap :");
		
			Set<Map.Entry<Integer,Double>> set = result.entrySet();		//_mkd_trying to iterate through graph map
			
			for(Map.Entry<Integer,Double> me : set)
			{
				Integer q = me.getKey();
				Double e = me.getValue();
				double cost = e;
				int cost1= (int) cost;
				if(e != null)
				{
					System.out.println(cost1);
				}
        	}

		}
		
//#####################################################################################################
    
	
		//Check for Simple Scheme in User mode     
	    else if(args[0].equals("-s"))
		{
			File file = new File(args[1]);  			//input file
			BufferedReader bufRdr = new BufferedReader(new FileReader(file));  
			//read from the text file
			String line = null;     					//initializing line
			int row = 0;        						//initialing row
			int col = 0;        						//initializing column
			int sNode=0;
			int numVert=0;
			int numEdge=0;
			
			
			if((line = bufRdr.readLine()) != null)
			{
				StringTokenizer st = new StringTokenizer(line," ");
				sNode = Integer.parseInt(st.nextToken());
				if((line = bufRdr.readLine()) != null)
				{
					StringTokenizer st1 = new StringTokenizer(line," ");
					numVert = Integer.parseInt(st1.nextToken());
					numEdge = Integer.parseInt(st1.nextToken());
				}
				else
				{
					System.out.print("looks like your file doesn't have second row, number of vertices and edges? ");
				}
			}
			
			else
			{
				System.out.println("looks like the file doesn't have valid value for source node");
			}
			
			//System.out.println("I got " + sNode + " as source node "+ numVert +" as number of vertices and "+numEdge+ " as number of Edges. Correct? ");
			int maxVert=((numVert *(numVert-1)) /2);
			double [][] arr = new double[maxVert][3];
			String [][] num1 = new String [maxVert][3];
			
			WeightedGraph grph_simple = new WeightedGraph (numVert);
			
			while((line = bufRdr.readLine()) != null && row<=(numEdge+2) )
			{ 
				StringTokenizer st = new StringTokenizer(line," ");
				while (st.hasMoreTokens())
				{
					num1[row][col] = st.nextToken(); 						//get next token and store it in the array
					arr[row][col]=Double.parseDouble(num1[row][col]);  		//store the data read from the file in an 2-dimentional array
																			//type casting the strings stored from the file to double
					col++;      											//increment the column counter
					
				}
				col = 0;    												//the first column of the next row
				row++;      												//next row
			}     
   
			bufRdr.close();
			
			for(int q=0;q<numVert;q++)					//populate graph by adding vertices
			{				
				grph.addNode(q);
				theGraph.addVertex(String.valueOf(q));
				grph_simple.setLabel(q,q);
			}

			for(int w=0;w<numEdge;w++)
			{
				/*add edges between vertices of the graph with the data from the file*/
				grph.addEdge(Integer.parseInt(num1[w][0]), Integer.parseInt(num1[w][1]), Double.parseDouble(num1[w][2]));
				theGraph.addEdge(Integer.parseInt(num1[w][0]),Integer.parseInt(num1[w][1]));
				grph_simple.addEdge(Integer.parseInt(num1[w][0]),Integer.parseInt(num1[w][1]),Integer.parseInt(num1[w][2]));
			}

			/*IMPLEMENTING DIJIKSTRA'S ALGORITHM USING SIMPLE SCHEME IN USER MODE*/
			//call Dijkstra_Two and print result
			int [] dist = Dijkstra_Two.dijkstra (grph_simple, sNode);
			System.out.println("Result for Dijkstra's algorithm using Simple Scheme: ");
			for(int i : dist)
				System.out.println(i);
			
		}

	}		// End of main Method
//#####################################################################################################
	
			public static int randomnumber(int a)		//creating random number 
		   {  
				Random rand = new Random();
				int q=(int)(a * rand.nextDouble());
				return q;
			}
	
	
}				//End of class Dijikstra

			
//#####################################################################################################
final class Dijkstra 
{
    /**
     * Given a directed, weighted graph G and a source node s, produces the
     * distances from s to each other node in the graph.  If any nodes in
     * the graph are unreachable from s, they will be reported at distance
     * +infinity.
     *
     * @param graph The graph upon which to run Dijkstra's algorithm.
     * @param source The source node in the graph.
     * @return A map from nodes in the graph to their distances from the source.
     */
    public static <T> Map<T, Double> shortestPaths(UndirectedGraph<T> graph, T source) {
        /* Create a Fibonacci heap storing the distances of unvisited nodes
         * from the source node.
         */
        FibonacciHeap<T> pq = new FibonacciHeap<T>();

        /* The Fibonacci heap uses an internal representation that hands back
         * Entry objects for every stored element.  This map associates each
         * node in the graph with its corresponding Entry.
         */
        Map<T, FibonacciHeap.Entry<T>> entries = new HashMap<T, FibonacciHeap.Entry<T>>();

        /* Maintain a map from nodes to their distances.  Whenever we expand a
         * node for the first time, we'll put it in here.
         */
        Map<T, Double> result = new HashMap<T, Double>();

        /* Add each node to the Fibonacci heap at distance +infinity since
         * initially all nodes are unreachable.
         */
        for (T node: graph)
            entries.put(node, pq.enqueue(node, Double.POSITIVE_INFINITY));

        /* Update the source so that it's at distance 0.0 from itself; after
         * all, we can get there with a path of length zero!
         */
        pq.decreaseKey(entries.get(source), 0.0);

        /* Keep processing the queue until no nodes remain. */
        while (!pq.isEmpty()) {
            /* Grab the current node.  The algorithm guarantees that we now
             * have the shortest distance to it.
             */
            FibonacciHeap.Entry<T> curr = pq.dequeueMin();

            /* Store this in the result table. */
            result.put(curr.getValue(), curr.getPriority());

            /* Update the priorities of all of its edges. */
            for (Map.Entry<T, Double> arc : graph.edgesFrom(curr.getValue()).entrySet()) {
                /* If we already know the shortest path from the source to
                 * this node, don't add the edge.
                 */
                if (result.containsKey(arc.getKey())) continue;

                /* Compute the cost of the path from the source to this node,
                 * which is the cost of this node plus the cost of this edge.
                 */
                double pathCost = curr.getPriority() + arc.getValue();

                /* If the length of the best-known path from the source to
                 * this node is longer than this potential path cost, update
                 * the cost of the shortest path.
                 */
                FibonacciHeap.Entry<T> dest = entries.get(arc.getKey());
                if (pathCost < dest.getPriority())
                    pq.decreaseKey(dest, pathCost);
            }
        }

        /* Finally, report the distances we've found. */
        return result;
    }
}




//#####################################################################################################

class Dijkstra_Two
{
 	
		// Dijkstra's algorithm to find shortest path from s to all other nodes
		public static int [] dijkstra (WeightedGraph G, int s) 
		{
		final int [] dist = new int [G.size()];  // shortest known distance from "s"
        final int [] pred = new int [G.size()];  // preceeding node in path
        final boolean [] visited = new boolean [G.size()]; // all false initially
   
        for (int i=0; i<dist.length; i++)
		{
           dist[i] = Integer.MAX_VALUE;
        }
        dist[s] = 0;
  
        for (int i=0; i<dist.length; i++) 
		{
           final int next = minVertex (dist, visited);
		   if (next == -1)
		  { System.out.print("This configuration doesn't give a connected graph, please try some other input");
		    System.exit(1);
		   }
		   //System.out.println("value of dist.length in line 298 "+dist.length);
           //System.out.println("value of variable next in line 306 "+next);
		   visited[next] = true;
  
           // The shortest path to next is dist[next] and via pred[next].
  
           final int [] n = G.neighbors (next);
           for (int j=0; j<n.length; j++) {
              final int v = n[j];
              final int d = dist[next] + G.getWeight(next,v);
              if (dist[v] > d) {
                 dist[v] = d;
                 pred[v] = next;
              }
           }
        }
  
		return dist;  // (ignore pred[s]==0!)
     }
  
     private static int minVertex (int [] dist, boolean [] v) {
        int x = Integer.MAX_VALUE;
        int y = -1;   // graph not connected, or no unvisited vertices
        for (int i=0; i<dist.length; i++) {
           if (!v[i] && dist[i]<x) {y=i; x=dist[i];}
        }
        return y;
     }
  
  }


//#####################################################################################################


 final class UndirectedGraph<T> implements Iterable<T> {
    /* A map from nodes in the graph to sets of outgoing edges.  Each
     * set of edges is represented by a map from edges to doubles.
     */
    private final Map<T, Map<T, Double>> mGraph = new HashMap<T, Map<T, Double>>();

    /**
     * Adds a new node to the graph.  If the node already exists, this
     * function is a no-op.
     *
     * @param node The node to add.
     * @return Whether or not the node was added.
     */
    public boolean addNode(T node) {
        /* If the node already exists, don't do anything. */
        if (mGraph.containsKey(node))
            return false;

        /* Otherwise, add the node with an empty set of outgoing edges. */
        mGraph.put(node, new HashMap<T, Double>());
			return true;
    }

    /**
     * Given a start node, destination, and length, adds an arc from the
     * start node to the destination of the length.  If an arc already
     * existed, the length is updated to the specified value.  If either
     * endpoint does not exist in the graph, throws a NoSuchElementException.
     *
     * @param start The start node.
     * @param dest The destination node.
     * @param length The length of the edge.
     * @throws NoSuchElementException If either the start or destination nodes
     *                                do not exist.
     */
    public void addEdge(T start, T dest, double length) {
        /* Confirm both endpoints exist. */
        if (!mGraph.containsKey(start) || !mGraph.containsKey(dest))
            throw new NoSuchElementException("Both nodes must be in the graph.");

        /* Add the edge. */
        mGraph.get(start).put(dest, length);
		mGraph.get(dest).put(start, length);
	}

    /**
     * Removes the edge from start to dest from the graph.  If the edge does
     * not exist, this operation is a no-op.  If either endpoint does not
     * exist, this throws a NoSuchElementException.
     *
     * @param start The start node.
     * @param dest The destination node.
     * @throws NoSuchElementException If either node is not in the graph.
     */
    public void removeEdge(T start, T dest) {
        /* Confirm both endpoints exist. */
        if (!mGraph.containsKey(start) || !mGraph.containsKey(dest))
            throw new NoSuchElementException("Both nodes must be in the graph.");

        mGraph.get(start).remove(dest);
		mGraph.get(dest).remove(start);
    }

    /**
     * Given a node in the graph, returns an immutable view of the edges
     * leaving that node, as a map from endpoints to costs.
     *
     * @param node The node whose edges should be queried.
     * @return An immutable view of the edges leaving that node.
     * @throws NoSuchElementException If the node does not exist.
     */
    public Map<T, Double> edgesFrom(T node) {
        /* Check that the node exists. */
        Map<T, Double> arcs = mGraph.get(node);
        if (arcs == null)
            throw new NoSuchElementException("Source node does not exist.");

        return Collections.unmodifiableMap(arcs);
    }

    /**
     * Returns an iterator that can traverse the nodes in the graph.
     *
     * @return An iterator that traverses the nodes in the graph.
     */
    public Iterator<T> iterator() {
        return mGraph.keySet().iterator();
    }
	
	
	
	
}

 //#####################################################################################################
	class Vertex 
	{
		public String label;       
		public boolean wasVisited;
		public Vertex(String lab)  
		{
			label = lab;
			wasVisited = false;
		}
	}
 
 //#####################################################################################################
 
 class Graph 
	{
		public int count=0;
		int size;
		private final int maxVertex = 6000;
		public Vertex vertexList[]; 						// list of vertices
		private int adjMat[][];      						// ADJACENCY MATRIX

		private int nVerts;          						// current number of vertices
		private Stack theStack;

		public Graph()               						// constructor
     {
			vertexList = new Vertex[maxVertex];
			adjMat = new int[maxVertex][maxVertex]; 		// ADJACENCY MATRIX
			nVerts = 0;
			for(int y=0; y<maxVertex; y++)      			// set adjacency
				for(int x=0; x<maxVertex; x++)   			// matrix to 0
					adjMat[x][y] = 0;
			theStack = new Stack();
     }  

		public void addVertex(String lab)
     {
			vertexList[nVerts++] = new Vertex(lab);
     }

		public void addEdge(int start, int end)
		{
			adjMat[start][end] = 1;
			adjMat[end][start] = 1;
		}

		public int dfs()  									// depth-first search
		{                                 					// begin at vertex 0
			vertexList[0].wasVisited = true;  				// mark it
			theStack.push(0);                 				// push it

			while( !theStack.isEmpty() )      				// until stack empty,
			{
				int v=-1;									// get an unvisited vertex adjacent to stack top
				int b = theStack.peek();
				for(int j=1; j<=nVerts; j++)
				{
					if(adjMat[b][j]==1 && vertexList[j].wasVisited==false)
					{ v=j;}
				}
          
				if(v == -1) 
				{                  							// if no such vertex,
					theStack.pop();
					count++;
				}
				else                          				// if it exists,
				{
       			vertexList[v].wasVisited = true;  		// mark it
					theStack.push(v);                 		// push it
				}
			}												// stack is empty, so we're done  

			
			for(int j=0; j<nVerts; j++)						// reset flags 
			{         
				vertexList[j].wasVisited = false;
			}
			
			return count;
  
		}  													// end of dfs
	}

//#####################################################################################################	
	
	class WeightedGraph 
 {
  
     private int [][]  edges;  // adjacency matrix
     private Object [] labels;
  
     public WeightedGraph (int n) 
	 {
        edges  = new int [n][n];
        labels = new Object[n];
     }
  
  
     public int size()
	 { 
		return labels.length; 
	 }
  
     public void   setLabel (int vertex, Object label) { labels[vertex]=label; }
     public Object getLabel (int vertex)               { return labels[vertex]; }
  
     public void    addEdge    (int source, int target, int w)  { edges[source][target] = w; }
     public boolean isEdge     (int source, int target)  { return edges[source][target]>0; }
     public void    removeEdge (int source, int target)  { edges[source][target] = 0; }
     public int     getWeight  (int source, int target)  { return edges[source][target]; }
  
     public int [] neighbors (int vertex) {
        int count = 0;
        for (int i=0; i<edges[vertex].length; i++) {
           if (edges[vertex][i]>0) count++;
        }
        final int[]answer= new int[count];
        count = 0;
        for (int i=0; i<edges[vertex].length; i++) {
           if (edges[vertex][i]>0) answer[count++]=i;
        }
        return answer;
     }
  
     public void print () {
        for (int j=0; j<edges.length; j++) {
           System.out.print (labels[j]+": ");
           for (int i=0; i<edges[j].length; i++) {
              if (edges[j][i]>0) System.out.print (labels[i]+":"+edges[j][i]+" ");
           }
           System.out.println ();
        }
     }
 }
	
//#####################################################################################################	
	
	class Stack 
	{
		private final int SIZE = 6000;
		private int[] st;
		private int top;

		public Stack()           // constructor
		{
			st = new int[SIZE];    // make array
			top = -1;
		}

		public void push(int j)   // put item on stack
		{ 
			st[++top] = j; 
		}

		public int pop()          // take item off stack
		{
			return st[top--]; 
		}

		public int peek()         // peek at top of stack
		{
			return st[top]; 
		}

		public boolean isEmpty()  // true if nothing on stack
		{
			return (top == -1); 
		}

	}
 
 
	//#####################################################################################################
	
	
final class FibonacciHeap<T> {
    /* In order for all of the Fibonacci heap operations to complete in O(1),
     * clients need to have O(1) access to any element in the heap.  We make
     * this work by having each insertion operation produce a handle to the
     * node in the tree.  In actuality, this handle is the node itself, but
     * we guard against external modification by marking the internal fields
     * private.
     */
    public static final class Entry<T> {
        private int     mDegree = 0;       // Number of children
        private boolean mIsMarked = false; // Whether this node is marked

        private Entry<T> mNext;   // Next and previous elements in the list
        private Entry<T> mPrev;

        private Entry<T> mParent; // Parent in the tree, if any.

        private Entry<T> mChild;  // Child node, if any.

        private T      mElem;     // Element being stored here
        private double mPriority; // Its priority

        /**
         * Returns the element represented by this heap entry.
         *
         * @return The element represented by this heap entry.
         */
        public T getValue() {
            return mElem;
        }
        /**
         * Sets the element associated with this heap entry.
         *
         * @param value The element to associate with this heap entry.
         */
        public void setValue(T value) {
            mElem = value;
        }

        /**
         * Returns the priority of this element.
         *
         * @return The priority of this element.
         */
        public double getPriority() {
            return mPriority;
        }

        /**
         * Constructs a new Entry that holds the given element with the indicated 
         * priority.
         */
        private Entry(T elem, double priority) {
            mNext = mPrev = this;
            mElem = elem;
            mPriority = priority;
        }
    }

    /* Pointer to the minimum element in the heap. */
    private Entry<T> mMin = null;

    /* Cached size of the heap, so we don't have to recompute this explicitly. */
    private int mSize = 0;

    /**
     * Inserts the specified element into the Fibonacci heap with the specified
     * priority.  Its priority must be a valid double, so you cannot set the
     * priority to NaN.
     */
    public Entry<T> enqueue(T value, double priority) {
        checkPriority(priority);

        /* Create the entry object, which is a circularly-linked list of length
         * one.
         */
        Entry<T> result = new Entry<T>(value, priority);

        /* Merge this singleton list with the tree list. */
        mMin = mergeLists(mMin, result);

        /* Increase the size of the heap; we just added something. */
        ++mSize;

        /* Return the reference to the new element. */
        return result;
    }

    /**
     * Returns an Entry object corresponding to the minimum element of the
     * Fibonacci heap, throwing a NoSuchElementException if the heap is
     * empty.
     */
    public Entry<T> min() {
        if (isEmpty())
            throw new NoSuchElementException("Heap is empty.");
        return mMin;
    }

    /**
     * Returns whether the heap is empty.
     *
     * @return Whether the heap is empty.
     */
    public boolean isEmpty() {
        return mMin == null;
    }

    /**
     * Returns the number of elements in the heap.
     *
     * @return The number of elements in the heap.
     */
    public int size() {
        return mSize;
    }

    /**
     * Given two Fibonacci heaps, returns a new Fibonacci heap that contains
     * all of the elements of the two heaps.  Each of the input heaps is
     * destructively modified by having all its elements removed.  You can
     * continue to use those heaps, but be aware that they will be empty
     * after this call completes.
     */
    public static <T> FibonacciHeap<T> merge(FibonacciHeap<T> one, FibonacciHeap<T> two) {
        /* Create a new FibonacciHeap to hold the result. */
        FibonacciHeap<T> result = new FibonacciHeap<T>();

        /* Merge the two Fibonacci heap root lists together.  This helper function
         * also computes the min of the two lists, so we can store the result in
         * the mMin field of the new heap.
         */
        result.mMin = mergeLists(one.mMin, two.mMin);

        /* The size of the new heap is the sum of the sizes of the input heaps. */
        result.mSize = one.mSize + two.mSize;

        /* Clear the old heaps. */
        one.mSize = two.mSize = 0;
        one.mMin  = null;
        two.mMin  = null;

        /* Return the newly-merged heap. */
        return result;
    }

    /**
     * Dequeues and returns the minimum element of the Fibonacci heap.  If the
     * heap is empty, this throws a NoSuchElementException.
     *
     * @return The smallest element of the Fibonacci heap.
     * @throws NoSuchElementException If the heap is empty.
     */
	 
		public Entry<T> dequeueMin() {
        /* Check for whether we're empty. */
        if (isEmpty())
            throw new NoSuchElementException("Heap is empty.");

        /* Otherwise, we're about to lose an element, so decrement the number of
         * entries in this heap.
         */
        --mSize;

        /* Grab the minimum element so we know what to return. */
        Entry<T> minElem = mMin;

        /* Now, we need to get rid of this element from the list of roots.  
         */
        if (mMin.mNext == mMin) { // Case one
            mMin = null;
        }
        else { // Case two
            mMin.mPrev.mNext = mMin.mNext;
            mMin.mNext.mPrev = mMin.mPrev;
            mMin = mMin.mNext; // Arbitrary element of the root list.
        }

        /* Next, clear the parent fields of all of the min element's children,
         * since they're about to become roots. 
         */
        if (minElem.mChild != null) {
            /* Keep track of the first visited node. */
            Entry<?> curr = minElem.mChild;
            do {
                curr.mParent = null;

                /* Walk to the next node, then stop if this is the node we
                 * started at.
                 */
                curr = curr.mNext;
            } while (curr != minElem.mChild);
        }

        /* Next, splice the children of the root node into the topmost list, 
         * then set mMin to point somewhere in that list.
         */
        mMin = mergeLists(mMin, minElem.mChild);

        /* If there are no entries left, we're done. */
        if (mMin == null) return minElem;

        /* Next, we need to coalsce all of the roots so that there is only one
         * tree of each degree. 
         */
        List<Entry<T>> treeTable = new ArrayList<Entry<T>>();

        // We need to traverse the entire list now
    
        List<Entry<T>> toVisit = new ArrayList<Entry<T>>();

        /* To add everything, we'll iterate across the elements until we
         * find the first element twice.  We check this by looping while the
         * list is empty or while the current element isn't the first element
         * of that list.
         */
        for (Entry<T> curr = mMin; toVisit.isEmpty() || toVisit.get(0) != curr; curr = curr.mNext)
            toVisit.add(curr);

        /* Traverse this list and perform the appropriate unioning steps. */
        for (Entry<T> curr: toVisit) {
            /* Keep merging until a match arises. */
            while (true) {
                /* Ensure that the list is long enough to hold an element of this
                 * degree.
                 */
                while (curr.mDegree >= treeTable.size())
                    treeTable.add(null);

                /* If nothing's here, we're can record that this tree has this size
                 * and are done processing.
                 */
                if (treeTable.get(curr.mDegree) == null) {
                    treeTable.set(curr.mDegree, curr);
                    break;
                }

                /* Otherwise, merge with what's there. */
                Entry<T> other = treeTable.get(curr.mDegree);
                treeTable.set(curr.mDegree, null); // Clear the slot

                /* Determine which of the two trees has the smaller root, storing
                 * the two tree accordingly.
                 */
                Entry<T> min = (other.mPriority < curr.mPriority)? other : curr;
                Entry<T> max = (other.mPriority < curr.mPriority)? curr  : other;

                /* Break max out of the root list, then merge it into min's child
                 * list.
                 */
                max.mNext.mPrev = max.mPrev;
                max.mPrev.mNext = max.mNext;

                /* Make it a singleton so that we can merge it. */
                max.mNext = max.mPrev = max;
                min.mChild = mergeLists(min.mChild, max);
                
                /* Reparent max appropriately. */
                max.mParent = min;

                /* Clear max's mark, since it can now lose another child. */
                max.mIsMarked = false;

                /* Increase min's degree; it now has another child. */
                ++min.mDegree;

                /* Continue merging this tree. */
                curr = min;
            }

            /* Update the global min based on this node.  Note that we compare
             * for <= instead of < here.  That's because if we just did a
             * reparent operation that merged two different trees of equal
             * priority, we need to make sure that the min pointer points to
             * the root-level one.
             */
            if (curr.mPriority <= mMin.mPriority) mMin = curr;
        }
        return minElem;
    }

    /**
     * Decreases the key of the specified element to the new priority.  If the
     * new priority is greater than the old priority, this function throws an
     * IllegalArgumentException.  The new priority must be a finite double,
     * so you cannot set the priority to be NaN, or +/- infinity.  Doing
     * so also throws an IllegalArgumentException.
     *
     * It is assumed that the entry belongs in this heap.  For efficiency
     * reasons, this is not checked at runtime.
     *
     * @param entry The element whose priority should be decreased.
     * @param newPriority The new priority to associate with this entry.
     * @throws IllegalArgumentException If the new priority exceeds the old
     *         priority, or if the argument is not a finite double.
     */
    public void decreaseKey(Entry<T> entry, double newPriority) {
        checkPriority(newPriority);
        if (newPriority > entry.mPriority)
            throw new IllegalArgumentException("New priority exceeds old.");

        /* Forward this to a helper function. */
        decreaseKeyUnchecked(entry, newPriority);
    }
    
    /**
     * Deletes this Entry from the Fibonacci heap that contains it.
     *
     * It is assumed that the entry belongs in this heap.  For efficiency
     * reasons, this is not checked at runtime.
     *
     * @param entry The entry to delete.
     */
    public void delete(Entry<T> entry) {
        /* Use decreaseKey to drop the entry's key to -infinity.  This will
         * guarantee that the node is cut and set to the global minimum.
         */
        decreaseKeyUnchecked(entry, Double.NEGATIVE_INFINITY);

        /* Call dequeueMin to remove it. */
        dequeueMin();
    }

    /**
     * Utility function which, given a user-specified priority, checks whether
     * it's a valid double and throws an IllegalArgumentException otherwise.
     *
     * @param priority The user's specified priority.
     * @throws IllegalArgumentException If it is not valid.
     */
    private void checkPriority(double priority) {
        if (Double.isNaN(priority))
            throw new IllegalArgumentException(priority + " is invalid.");
    }


    private static <T> Entry<T> mergeLists(Entry<T> one, Entry<T> two) {
        /* There are four cases depending on whether the lists are null or not.
         * We consider each separately.
         */
        if (one == null && two == null) { // Both null, resulting list is null.
            return null;
        }
        else if (one != null && two == null) { // Two is null, result is one.
            return one;
        }
        else if (one == null && two != null) { // One is null, result is two.
            return two;
        }
        else 
		{ // Both non-null; actually do the splice.
            
            Entry<T> oneNext = one.mNext; // Cache this since we're about to overwrite it.
            one.mNext = two.mNext;
            one.mNext.mPrev = one;
            two.mNext = oneNext;
            two.mNext.mPrev = two;

            /* Return a pointer to whichever's smaller. */
            return one.mPriority < two.mPriority? one : two;
        }
    }

    /**
     * Decreases the key of a node in the tree without doing any checking to ensure
     * that the new priority is valid.
     *
     * @param entry The node whose key should be decreased.
     * @param priority The node's new priority.
     */
    private void decreaseKeyUnchecked(Entry<T> entry, double priority) {
        /* First, change the node's priority. */
        entry.mPriority = priority;

        
        if (entry.mParent != null && entry.mPriority <= entry.mParent.mPriority)
            cutNode(entry);

        
        if (entry.mPriority <= mMin.mPriority)
            mMin = entry;
    }


    private void cutNode(Entry<T> entry) {
        /* Begin by clearing the node's mark, since we just cut it. */
        entry.mIsMarked = false;

        /* Base case: If the node has no parent, we're done. */
        if (entry.mParent == null) return;

        /* Rewire the node's siblings around it, if it has any siblings. */
        if (entry.mNext != entry) { // Has siblings
            entry.mNext.mPrev = entry.mPrev;
            entry.mPrev.mNext = entry.mNext;
        }

        if (entry.mParent.mChild == entry) {
            /* If there are any other children, pick one of them arbitrarily. */
            if (entry.mNext != entry) {
                entry.mParent.mChild = entry.mNext;
            }

            else {
                entry.mParent.mChild = null;
            }
        }

        /* Decrease the degree of the parent, since it just lost a child. */
        --entry.mParent.mDegree;

        /* Splice this tree into the root list by converting it to a singleton
         * and invoking the merge subroutine.
         */
        entry.mPrev = entry.mNext = entry;
        mMin = mergeLists(mMin, entry);

        /* Mark the parent and recursively cut it if it's already been
         * marked.
         */
        if (entry.mParent.mIsMarked)
            cutNode(entry.mParent);
        else
            entry.mParent.mIsMarked = true;

        /* Clear the relocated node's parent; it's now a root. */
        entry.mParent = null;
    }
}