# Assignment 1

### Python Version

* 2.7.10

## Running the Program

```
python puzzlesolver.py filename algorithm_name heurisitic_func_name(optional)
```

Other than the heuristic function, all other keywords are followed the same way as given in the file.

### Heuristic function names for different puzzles

#### Water Jug

* *goal_test* 
* *absolute_difference*

#### Path Planning

* *manhattan_dist* 
* *euclidean_dist*
* *goal_test*

#### Burnt Pancake

* *misplaced_num* 
* *goal_test*

## References

* http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
* https://gist.github.com/jamiees2/5527632
* http://stackoverflow.com/questions/26146342/how-can-i-define-a-heuristic-function-for-water-jug
* http://cyluun.github.io/blog/uninformed-search-algorithms-in-python
* Python docs

## Code portion not working

* The dfs function is undergoing infinite loop even after performing checks whether the node is already present in frontier. However, the dfs algorithm used in itself is correct.
* Burnt_pancake puzzle time and space complexity is outside the scope of local machines.
