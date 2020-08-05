

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

$$ \max_{\mathbf{x}} \;\; \mathbf{r}^T \mathbf{x} - \gamma \mathbf{x}^T \mathbf{Q} \mathbf{x},  \;\;\;\;\;\;\;(1)$$

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
 \end{aligned} \;\;\;\;\;\;\; (2)$$
 

 Here it is important to note that our final constraint in equation 2 denotes that the elements of our player vector $\mathbf{x}$ are binary - meaning we can either have a player or we cannot, and $n$ is the size of this vector - the number of players in the NHL. With this in mind, the other constraints are a statement that say that we must chose the non-zero components of our player vector such that the sums of those elements which represent a player in those positions must satisfy the above conditions. Equation 2 represents what is known as a [binary integer programing](http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf) problem (of course, equation (1) and (2) are in quadratic form, but we will return to this later). Methodologies to solve questions of this nature will be discussed at length in later sections of this document, but for now, let us trust that solvers for this class of problem exists, and that we can implement them easily.

What equation 2 represents is a formal statement that the solution to the maximization problem should yield an "optimal" team, up to some risk parameter, which should see the maximum returns with respect to fantasy points.

## Two Brief Digressions 
 ### A Note On Returns

 You may have noticed that we have mentioned that the solution to equations 1 and 2 both rely on us to be able to have some metric $\mathbf{r}$ which represents the expected returns of a player, and not only that, to have enough data or an alternate methodology with which to calculate the covariance matrix $\mathbf{Q}$. In this case, finding the value for expected returns is largely an art form on its own - a more successful model than the one demonstrated here would also take advantage of expert advice and other insights to have the greatest estimate of returns and covariance. In our case however, we are data scientists and want to see what the data itself can tell us. As such, we will use the data right from the NHL of a players actions in each game and calculate their fantasy points. From there, we will use this set of all their points and simply work with the mean returns for $\mathbf{r}$ - how well the player performs on average. As we will also have all the data for each player - we will also be able to calculate their covariance matrix $\mathbf{Q}$ readily. 

 This is far from the best solution, and a more sports minded group would invest a great deal of time into finding the "perfect" estimates of the returns and covariance to also incorporate subject matter expertise, but from an exploration and education point of view, this approach is at least reasonable (well... to us... sports layman's), and has the advantage that we don't have to invest too much time in order to calculate this. Certainly - we can always revisit the returns vectors and covariance once we have a working solution. 

 However, to define these quantities formally, we have for the $i^{th}$ player, its element in the returns vector r_i is

 $$ r_i= \frac{1}{N}\sum_{j=1}^{N} p_j$$

 where $N$ is the number of games played, and the the element $p_j$ is the points that player earned in the $j^{th}$ game. Using this notation, our covariance matrix $\mathbf{Q}$ is calculated as

 $$ \mathbf{Q} = \frac{1}{N - 1} \sum_{i=0}^N \left(\mathbf{P} -  \mathbf{I \cdot r} \right) \left(\mathbf{P} -  \mathbf{I \cdot r} \right)^T$$

 where $\mathbf{P}$ is a rectangular matrix of size $h \times N$, where $h$ is the number of players, and is simply the "points matrix", or the table of points each player got in each game, and $\mathbf{I}$ is the identity matrix. 

 What is important to note howeveer in this instance is that _order is important_ . Here we have chosen to index by game number - but an important improvement could be made by indexing this matrix by each game played. This would allow us to view correlations between players across teams more accurately, and is a future step in this project. 

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
 \end{aligned} \;\;\;\;\;\;\; (3)$$

Where here $T_c$ is the set of players in our team that we have already chosen, and must be a part of our team, and $O_c$ is the set of players chosen by others that we can no longer pick, and $\alpha$ is an integer representing the round of the draft we are in. Using for notational convenience, we will denote our new constraints 

$$ \sum_{i \in T_c} \mathbf{x}_i = \alpha \;\;\;\;\;\;\; (4)$$ 
$$ \sum_{i \in O_c} \mathbf{x}_i = 0 \;\;\;\;\;\;\; (5)$$


so we may reference them directly in algorithm (1) below. 

**Algorithm(**_draft_**):**           (1)

---

 1. Choose a value for risk tolerance $\gamma$
 2.  Decide on team size **_MAX_**
 3. **_WHILE $\alpha \neq$ MAX_ DO:**
    * Update the set <img src="/tex/4221397c8a5a02a9d784666f47094f17.svg?invert_in_darkmode&sanitize=true" align=middle width=68.74498784999999pt height=24.65753399999998pt/> of players which have been drafted
        * Update constraint (5)
    * Solve the maximization problem (3)
        * Using some metric, choose a single player <img src="/tex/a9181dcbb0c785f87c807a62fbca43d5.svg?invert_in_darkmode&sanitize=true" align=middle width=14.15517674999999pt height=14.611878600000017pt/> from the solution of (3)
    * Update the set <img src="/tex/37333fbdc33aa49b7a5c0fb483c5d522.svg?invert_in_darkmode&sanitize=true" align=middle width=66.51064694999998pt height=24.65753399999998pt/> by adding the player we have chosen to our team
    * Allow other actors to choose their players.
    * **_IF $\alpha$ = MAX_**
        * **END**
    * **_Else_**
        * $\alpha = \alpha + 1$, update constraint equation (5). 
---

Where algorithm 1 requires us to solve the optimization problem (3) **_MAX_** times. This may seem wasteful as we're only choosing a single player each time, why solve the entire problem and generate what is essentially a new team every time? 

The answer to that is not as straightforward as you may think, but primarily it is because   a Markowitz style portfolio optimization is optimizing the entire set - we're looking for the players that will have the highest returns on average when they're "working together" that is - our optimal solution depends on our entire team, not just a single player.

Of course, we notice in step 2 of the main loop of algorithm (1) we have left unspecified _how_ to choose the single player that we're keeping in this round of the draft. Of course, this is another one of those stages where we need to decide for ourselves how we may choose a single player to include in our team out of the entire optimal team we have in the entire process. That is discussed in the next section

### Choosing A Single Player

There are many metrics by which we can choose a single player and add them to our teams set of players $T_C$ for each round of the draft, and not all of them are created equal but worth exploring. In fact, many of the fantasy teams we created for this project take advantage of different metrics with respect to their choosing criteria. For example, there are the two easiest choices for choosing a single player:

1. Choose an available player $x_i$ whose average returns $r_i$ are highest from the solution of equation (3). 
2. Choose an available player $x_i$ whose RMS of returns is highest from the solution of equation (3)

These two options are the easiest to implement and worth exploring, however, with all the computational power at hand - we may also be interested in trying to optimize once again in order to find the best player to choose. For example, if we again use a Markowitz style optimization using only the players from our specific team $T_{C_k}$, where the $k$ subscript indicates that this is the $k^{th}$ round of the draft, we could write the following optimization problem

$$\underset{\mathbf{x}}{\text{argmax}} \left\{\; \max_\mathbf{x} \; \; \mathbf{r}^T \mathbf{x} -\beta \mathbf{x}^T \mathbf{Q} \mathbf{x} \right\}_{\mathbf{x,r, Q} \in T_{C_k}}. \;\;\;\;\;\; (6)$$

Where in equation (6) above, the player vector $\mathbf{x}$, the returns vector $\mathbf{r}$ and covariance matrix $\mathbf{Q}$ are limited to only the entries which are relevant to the players in the current optimal team from the solution of equation (3) in draft round $k$, and $\beta$ is again a risk avoidance parameter. In the equation above, $\mathbf{x}$ is no longer discrete and we will return a player vector with real arguments. In this sense, the solution to the first maximization problem gives tells us how much to "invest" in a particular player for the optimal portfolio. The argument then, is if we can only pick one player, we want to pick the player with the highest investment, which is where the $\text{argmax}$ comes in. This is simply a statement that we will pick the player whose investment is greatest from the solution vector to our optimization problem. 

### Details On The Sub Optimization
Coming soon


## Binary Integer Programing
Coming soon. 

## SportsNet Fantasy Hockey

In this case, we have a slightly different problem to solve. Rather than solving several instances of an optimization problem using limited resources between players, here everyone is free to choose whatever player they want - subject to division, position, and a points value cap. Formally, 

$$\begin{aligned}
\text{maximize} & \;\; \mathbf{r}^T \mathbf{x} -\gamma \mathbf{x}^T \mathbf{Q} \mathbf{x} \\
\text{subject to} & \;\; \sum \mathbf{C} \cdot \mathbf{x} \leq 30 \\
& \sum_{\text{east goalies}} \mathbf{x} = \sum_{\text{west goalies}} \mathbf{x} = 2 \\
& \sum_{\text{east forwards}} \mathbf{x} = \sum_{\text{west forwards}} \mathbf{x} = 3 \\ 
& \sum_{\text{east defence}} \mathbf{x} = \sum_{\text{west defence}} \mathbf{x} = 2 \\
& \mathbf{x} \in \mathbb{B}
\end{aligned} $$

where $\mathbf{x}$ is a binary vector of players, $\mathbf{r}$ is our returns vector, $\mathbf{Q}$ is the covariance matrix, $\gamma$ is the risk avoidance parameter and $\mathbf{C}$ is a diagonal matrix of the cost, or points value assigned by SportsNet, associated with each player.

For the SportsNet contest, each player has a value of 1-4, and we have to assemble a team using 30 points or less, as well, we need to make sure we have players from each conference - represented in the additional constraints. Besides those new additions, this is actually an "easier" problem than the draft as we only have to choose one team (and then hope for the best). One thing that should be noted however, is that in this contest the point value system is different. Here, the points are described in the table below 

> Table 2: here is the point value system used to score each player in the Sportsnet fantasy draft

|                    | Goals | Assists | Wins | Shutout |
|--------------------|-------|---------|------|---------|
| Centers            |     1 |       1 |  N/A |     N/A |
| Left/Right Wingers |     1 |       1 |  N/A |     N/A |
| Defence            |     1 |       1 |  N/A |     N/A |
| Goalies            |   N/A |     N/A |    2 |       1 |

Here we see that goalies are really only rewarded for winning a game, and a little extra if they win, as compared to our previous example.

### Caveats for Playoff Hockey

In this case, as playoffs are elimination based, it is not favorable for us to choose players from teams that we may expect to lose. In this case, we introduced an artificial _win bias_ into the points scoring system during the optimization by awarding each player an additional 1.5 points for each game they have won. The idea here is that our optimization will now be biased towards teams that tend to win more games, which is ideal for playoff hockey. What should be noted is that formally, we're updating our points vector for each player $\mathbf{p}$ as 

$$\mathbf{p}_{\text{win bias}} = \mathbf{p} + \mathbf{b}$$

where $\mathbf{b}$ is a _bias_ vector with elements defined by

$$b_i = p_i + 1.5 \; \delta_w$$

where $\delta_w$ is the Kroneker delta, defined as 

$$\delta_w = \left\{ \begin{aligned} 1 & \text{ if} & \text{win}  \\ 0 & \text{ if} & \text{loss} \end{aligned} \right.$$

Where our calculations of our mean returns vector and covariance terms are identical, however, now we are using our win-biased points instead. 
