# FantasyHockey
Location of our Fantasy Hockey draft code to teach data science.

# Work In Progress
Stay tuned! We will be updating this with our methodologies as well as updates as we get to them!

# Road map for future

1. Implementation of "live" team trading and maintenance so the robots can drop/add players as they see fit with a stream of data
2. Implementation of a version of this where we take subject matter expert opinion into account as well
3. Implementation of a version using machine learning rather than the current classical approach of constrained optimization 
4. Implementation of our own player performance estimations 
5. Add more human players


# How It Works

In this repository we make use of [Markowitz Portfolio Theory](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf) in order to select the best "portfolio" of players at a given time. In the sections below we will follow the following notational conventions
1. Vectors will be shown in lower-case bold latin characters
2. Matrices will be shown in upper-case latin figures
3. Constants will be shown with greek characters.

Markowitz portfolio theory defines the optimal portfolio to be the solution to 

$$ \max_{\mathbf{x}} \;\; \mathbf{r}^T \mathbf{x} - \gamma \mathbf{x}^T \mathbf{Q} \mathbf{x},  \tag {1}$$

where $\mathbf{r}$ is the returns vector, $\mathbf{x}$ is the total set of assets, and $\mathbf{Q}$ is the covariance matrix of returns, and $\gamma$ is the "risk aversion constant" - a parameter that represents how much risk we are willing to accept. A $\gamma$ value of 0 would say that we do not tolerate any risk, and large values of $\gamma$ imply that we're willing to ["risk it to get the the biscuit"](https://www.youtube.com/watch?v=jz9uqs_IydY) In the context of finding the optimal portfolio of hockey players to make the fantasy hockey team - the returns $\mathbf{r}$ are measured in our expectation of how many points we expect particular player to get in a given game, and the covariance matrix $\mathbf{Q}$ is the covariance of all these expected points. As well, the vector $\mathbf{x}$ is the set of players which maximize our expected returns (fantasy points). 

In the scope of maximization problems like in equation 1, we also have to consider that the set of players $\mathbf{x}$ that we can choose is also subject to some _constraints_; we are limited in how we can pick our players. We need to ensure that we choose the correct amount of players to fill each position. For example, we need to ensure that we have enough (and not too many!) centers, wingers, defence men and goalies. To ensure this - we have to move from the unconstrained problem in equation 1 to a problem in constrained optimization. 

In this case, the choices may change from fantasy league to fantasy league - but in this example the table below lays out the constraints in how we may choose players for our fantasy teams 

|    **Position**    | **Minimum** | **Maximum** |
|:------------------:|:-----------:|:-----------:|
| Centers            |           2 |         N/A |
| Left/Right Wingers |         2/2 |   N/A / N/A |
| Defence            |           4 |         N/A |
| Goalies            |           2 |           3 |
| **Team Size**      |             |          17 |

Where from the table above, we notice that each team consists of 7 players, we need between 2 and 3 goalies, and we are required to have at least 2 centers, two of each left and right wingers, and 4 defence players. Mathematically with equation 1 we have,  where $\text{G, LW, RW, C, D, T}$ stand for Goalies, Left Wingers, Right Wingers, Centers, Defence and Team respectively.

$$\begin{aligned}
 \max_{\mathbf{x}} & \;\; \mathbf{r}^T \mathbf{x} - \gamma \mathbf{x}^T \mathbf{Q} \mathbf{x} \\ 
 \text{subject to} & \;\; 2 \leq \sum_{i \in \text{G}} \mathbf{x_i} \leq 3 \\
 & \;\; \sum_{i \in \text{LW}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{RW}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{C}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{D}} \mathbf{x_i} \geq 4 \\
 & \;\; \sum_{i \in \text{T}} \mathbf{x_i} = 17 \\
 & \;\; \mathbf{x} \in \left\{0 , 1 \right\}^n.
 \end{aligned} \tag {2}$$
 

 Here it is important to note that our final constraint in equation 2 denotes that the elements of our player vector $\mathbf{x}$ are binary - meaning we can either have a player or we cannot, and $n$ is the size of this vector - the number of players in the NHL. With this in mind, the other constraints are a statement that say that we must chose the non-zero components of our player vector such that the sums of those elements which represent a player in those positions must satisfy the above conditions. Equation 2 represents what is known as a [binary integer programing](http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf) problem. Methodologies to solve questions of this nature will be discussed at length in later sections of this document, but for now, let us trust that solvers for this class of problem exists, and that we can implement them easily.

What equation 2 represents is a formal statement that the solution to the maximization problem should yield an "optimal" team, up to some risk parameter, which should see the maximum returns with respect to fantasy points.

## Two Brief Digressions 
 ### A Note On Returns

 You may have noticed that we have mentioned that the solution to equations 1 and 2 both rely on us to be able to have some metric $\mathbf{r}$ which represents the expected returns of a player, and not only that, to have enough data or an alternate methodology with which to calculate the covariance matrix $\mathbf{Q}$. In this case, finding the value for expected returns is largely an art form on its own - a more successful model than the one demonstrated here would also take advantage of expert advice and other insights to have the greatest estimate of returns and covariance. In our case however, we are data scientists and want to see what the data itself can tell us. As such, we will use the data right from the NHL of a players actions in each game and calculate their fantasy points. From there, we will use this set of all their points and simply work with the mean returns for $\mathbf{r}$ - how well the player performs on average. As we will also have all the data for each player - we will also be able to calculate their covariance matrix $\mathbf{Q}$ readily. 

 This is far from the best solution, and a more sports minded group would invest a great deal of time into finding the "perfect" estimates of the returns and covariance to also incorporate subject matter expertise, but from an exploration and education point of view, this approach is at least reasonable (well... to us... sports layman's), and has the advantage that we don't have to invest too much time in order to calculate this. Certainly - we can always revisit the returns vectors and covariance once we have a working solution. 

 ### Dimensionality and Missing Data

 One subtlety of the above approach is that in order to calculate covariance each player needs to play the same amount of games. In other words, we need to have 82 data points (the number of games in an NHL season) for each player we wish to include, for each season that we're including them. This is a bit of a problem, as it is incredibly rare for a player to play every game in a season. So this leaves us with an important question:

 > How do we deal with missing data?

In this case, much like when choosing how to measure returns, we also need to choose how to deal with this missing data. In a perfect world, we would "impute" these values - essentially sampling a players distribution and filling in missing values with data that should be statistically relevant. However, this approach has a few problems, one of them being very technical, and one being very practical.

The technical problem is of course, we have no idea how to properly sample this distribution. How do we sample this distribution? Certainly players do not act independently - a proper sampling would require a multidimensional distribution that treats the relationships between players, particularity those on the same line, and perhaps rivals appropriately for a realistic sample. On the practical side of things, we're looking for a set of players which gives us maximum returns - and a player that did not play gave us exactly zero returns. So, if a player doesn't play often - that's important to know. As we're interested in maximizing returns, we have **chosen** to zero fill all returns for every player when they have missed a game. 

In this case, each player will have exactly 82 returns "measurements" throughout the season, set by us to zero for games they did not play. These measurements will allow us to both calculate our returns vectors $\mathbf{r}$ and our covariance matrix $\mathbf{Q}$

## The Draft

Equation 2 represents how to find the optimal team with no competition in a draft scenario - you will choose the best possible team because you can pick who you want, when you want. Unfortunately - this is not the case in a draft scenario. You get to pick one person at a time, and if someone else takes a player you're interested? Hope you have a back up plan. How do we change equation 2 such that we can account for this? 

Well, it turns out that this is more difficult to write in a closed form, and we now have to create an _algorithm_ that we can use to adjust and adapt to other people taking from our discrete set of players $\mathbf{x}$ before we get the chance. Our first step is to modify our optimization equation to account for two more conditions as follows

$$\begin{aligned}
 \max_{\mathbf{x}} & \;\; \mathbf{r}^T \mathbf{x} - \gamma \mathbf{x}^T \mathbf{Q} \mathbf{x} \\ 
 \text{subject to} & \;\; 2 \leq \sum_{i \in \text{G}} \mathbf{x_i} \leq 3 \\
 & \;\; \sum_{i \in \text{LW}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{RW}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{C}} \mathbf{x_i} \geq 2 \\
 & \;\; \sum_{i \in \text{D}} \mathbf{x_i} \geq 4 \\
 & \;\; \sum_{i \in \text{T}} \mathbf{x_i} = 17 \\
 & \;\; \mathbf{x} \in \left\{0 , 1 \right\}^n \\
 & \;\; \sum_{i \in T_c} \mathbf{x}_i = \alpha \\
 & \;\; \sum_{i \in O_c} \mathbf{x}_i = 0.
 \end{aligned} \tag {3}$$

Where here $T_c$ is the set of players in our team that we have already chosen, and must be a part of our team, and $O_c$ is the set of players chosen by others that we can no longer pick, and $\alpha$ is an integer representing the round of the draft we are in. Using for notational convenience, we will denote our new constraints 

$$ \sum_{i \in T_c} \mathbf{x}_i = R \tag{4}$$
$$ \sum_{i \in O_c} \mathbf{x}_i = 0 \tag{5}$$


so we may reference them directly in algorithm (1) below. 

**Algorithm(**_draft_**):**           (1)

---

1. Choose a value for risk tolerance $\gamma$
2. Decide on team size **_MAX_**
3. **_WHILE $\alpha \neq$ MAX_**
    1. Update the set $\left\{\mathbf{x}_i\right\}_{x_i \in O_c}$ of players which have been drafted
        * Update constraint (5)
    2. Solve the maximization problem (3)
        * Using some metric, choose a single player $\mathbf{x_i}$ from the solution of (3)
        * Update the set $\left\{\mathbf{x}_i\right\}_{x_i \in T_c}$ by adding the player we have chosen to our team
    3. Allow other actors to choose their players.
    3. **_IF $\alpha$ = MAX_**
        * **END**
    3. **_Else_**
        * $\alpha = \alpha + 1$, update constraint equation (5). 
---

Where algorithm 1 requires us to solve the optimization problem (3) **_MAX_** times. This may seem wasteful as we're only choosing a single player each time, why solve the entire problem and generate what is essentially a new team every time? 

The answer to that is not as straightforward as you may think, but primarily it is because   a Markowitz style portfolio optimization is optimizing the entire set - we're looking for the players that will have the highest returns on average when they're "working together" that is - our optimal solution depends on our entire team, not just a single player.

Of course, we notice in step 2 of the main loop of algorithm (1) we have left unspecified _how_ to choose the single player that we're keeping in this round of the draft. Of course, this is another one of those stages where we need to decide for ourselves how we may choose a single player to include in our team out of the entire optimal team we have in the entire process. That is discussed in the next section

### Choosing A Single Player


## Binary Integer Programing
Coming soon. 

