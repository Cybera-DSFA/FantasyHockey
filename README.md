

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

<p align="center"><img src="/tex/598eac94d582e1709bcd011d00e99fd0.svg?invert_in_darkmode&sanitize=true" align=middle width=204.8825724pt height=24.5133075pt/></p>

where <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> is the returns vector, <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> is the total set of assets, and <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> is the covariance matrix of returns, and <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is the "risk aversion constant" - a parameter that represents how much risk we are willing to accept. A <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> value of 0 would say that we do not tolerate any risk, and large values of <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> imply that we're willing to ["risk it to get the the biscuit"](https://www.youtube.com/watch?v=jz9uqs_IydY) In the context of finding the optimal portfolio of hockey players to make the fantasy hockey team - the returns <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> are measured in our expectation of how many points we expect particular player to get in a given game, and the covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> is the covariance of all these expected points. As well, the vector <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> is the set of players which maximize our expected returns (fantasy points). 

In the scope of maximization problems like in equation 1, we also have to consider that the set of players <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> that we can choose is also subject to some _constraints_; we are limited in how we can pick our players. We need to ensure that we choose the correct amount of players to fill each position. For example, we need to ensure that we have enough (and not too many!) centers, wingers, defence men and goalies. To ensure this - we have to move from the unconstrained problem in equation 1 to a problem in constrained optimization. 

In this case, the choices may change from fantasy league to fantasy league - but in this example the table below lays out the constraints in how we may choose players for our fantasy teams 

|    **Position**    | **Minimum** | **Maximum** |
|:------------------:|:-----------:|:-----------:|
| Centers            |           2 |         N/A |
| Left/Right Wingers |         2/2 |   N/A / N/A |
| Defence            |           4 |         N/A |
| Goalies            |           2 |           3 |
| **Team Size**      |             |          17 |

Where from the table above, we notice that each team consists of 7 players, we need between 2 and 3 goalies, and we are required to have at least 2 centers, two of each left and right wingers, and 4 defence players. Mathematically with equation 1 we have,  where <img src="/tex/d17baa61b3d37a4664e8b5a400ad1733.svg?invert_in_darkmode&sanitize=true" align=middle width=151.94089019999998pt height=22.465723500000017pt/> stand for Goalies, Left Wingers, Right Wingers, Centers, Defence and Team respectively.

<p align="center"><img src="/tex/86f142f7f3c0eef5c052208529001e99.svg?invert_in_darkmode&sanitize=true" align=middle width=236.02424175pt height=326.6158104pt/></p>
 

 Here it is important to note that our final constraint in equation 2 denotes that the elements of our player vector <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> are binary - meaning we can either have a player or we cannot, and <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the size of this vector - the number of players in the NHL. With this in mind, the other constraints are a statement that say that we must chose the non-zero components of our player vector such that the sums of those elements which represent a player in those positions must satisfy the above conditions. Equation 2 represents what is known as a [binary integer programing](http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf) problem (of course, equation (1) and (2) are in quadratic form, but we will return to this later). Methodologies to solve questions of this nature will be discussed at length in later sections of this document, but for now, let us trust that solvers for this class of problem exists, and that we can implement them easily.

What equation 2 represents is a formal statement that the solution to the maximization problem should yield an "optimal" team, up to some risk parameter, which should see the maximum returns with respect to fantasy points.

## Two Brief Digressions 
 ### A Note On Returns

 You may have noticed that we have mentioned that the solution to equations 1 and 2 both rely on us to be able to have some metric <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> which represents the expected returns of a player, and not only that, to have enough data or an alternate methodology with which to calculate the covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/>. In this case, finding the value for expected returns is largely an art form on its own - a more successful model than the one demonstrated here would also take advantage of expert advice and other insights to have the greatest estimate of returns and covariance. In our case however, we are data scientists and want to see what the data itself can tell us. As such, we will use the data right from the NHL of a players actions in each game and calculate their fantasy points. From there, we will use this set of all their points and simply work with the mean returns for <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> - how well the player performs on average. As we will also have all the data for each player - we will also be able to calculate their covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> readily. 

 This is far from the best solution, and a more sports minded group would invest a great deal of time into finding the "perfect" estimates of the returns and covariance to also incorporate subject matter expertise, but from an exploration and education point of view, this approach is at least reasonable (well... to us... sports layman's), and has the advantage that we don't have to invest too much time in order to calculate this. Certainly - we can always revisit the returns vectors and covariance once we have a working solution. 

 However, to define these quantities formally, we have for the <img src="/tex/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32504519999999pt height=27.91243950000002pt/> player, its element in the returns vector r_i is

 <p align="center"><img src="/tex/45048b77d5af9f1b01e28379bcfec173.svg?invert_in_darkmode&sanitize=true" align=middle width=97.35062699999999pt height=50.04352485pt/></p>

 where <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is the number of games played, and the the element <img src="/tex/7f131a60c8e7bb2b22f383f7bd49e2c0.svg?invert_in_darkmode&sanitize=true" align=middle width=14.37507554999999pt height=14.15524440000002pt/> is the points that player earned in the <img src="/tex/95291b39ba5d9dba052b40bf07b12cd2.svg?invert_in_darkmode&sanitize=true" align=middle width=20.37223649999999pt height=27.91243950000002pt/> game. Using this notation, our covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> is calculated as

 <p align="center"><img src="/tex/0a7e2d76cad69ff4714c522f3b9f45b3.svg?invert_in_darkmode&sanitize=true" align=middle width=270.12137624999997pt height=47.806078649999996pt/></p>

 where <img src="/tex/384591906555413c452c93e493b2d4ec.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92230829999999pt height=22.55708729999998pt/> is a rectangular matrix of size <img src="/tex/0d5fa05b4ddb8e121849096175e1d42c.svg?invert_in_darkmode&sanitize=true" align=middle width=44.56229579999999pt height=22.831056599999986pt/>, where <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> is the number of players, and is simply the "points matrix", or the table of points each player got in each game, and <img src="/tex/d8471e559d932f20f66bec32f6002e08.svg?invert_in_darkmode&sanitize=true" align=middle width=7.168923299999991pt height=22.55708729999998pt/> is the identity matrix. 

 What is important to note howeveer in this instance is that _order is important_ . Here we have chosen to index by game number - but an important improvement could be made by indexing this matrix by each game played. This would allow us to view correlations between players across teams more accurately, and is a future step in this project. 

 ### Dimensionality and Missing Data

 One subtlety of the above approach is that in order to calculate covariance each player needs to play the same amount of games. In other words, we need to have 82 data points (the number of games in an NHL season) for each player we wish to include, for each season that we're including them. This is a bit of a problem, as it is incredibly rare for a player to play every game in a season. So this leaves us with an important question:

 > How do we deal with missing data?

In this case, much like when choosing how to measure returns, we also need to choose how to deal with this missing data. In a perfect world, we would "impute" these values - essentially sampling a players distribution and filling in missing values with data that should be statistically relevant. However, this approach has a few problems, one of them being very technical, and one being very practical.

The technical problem is of course, we have no idea how to properly sample this distribution. How do we sample this distribution? Certainly players do not act independently - a proper sampling would require a multidimensional distribution that treats the relationships between players, particularity those on the same line, and perhaps rivals appropriately for a realistic sample. On the practical side of things, we're looking for a set of players which gives us maximum returns - and a player that did not play gave us exactly zero returns. So, if a player doesn't play often - that's important to know. As we're interested in maximizing returns, we have **chosen** to zero fill all returns for every player when they have missed a game. 

In this case, each player will have exactly 82 returns "measurements" throughout the season, set by us to zero for games they did not play. These measurements will allow us to both calculate our returns vectors <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> and our covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/>

## The Draft

Equation 2 represents how to find the optimal team with no competition in a draft scenario - you will choose the best possible team because you can pick who you want, when you want. Unfortunately - this is not the case in a draft scenario. You get to pick one person at a time, and if someone else takes a player you're interested? Hope you have a back up plan. How do we change equation 2 such that we can account for this? 

Well, it turns out that this is more difficult to write in a closed form, and we now have to create an _algorithm_ that we can use to adjust and adapt to other people taking from our discrete set of players <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> before we get the chance. Our first step is to modify our optimization equation to account for two more conditions as follows

<p align="center"><img src="/tex/b8b231027c57b4dde2749366915271ca.svg?invert_in_darkmode&sanitize=true" align=middle width=236.02424175pt height=420.3148917pt/></p>

Where here <img src="/tex/5e868a5b4bdf4b99dde255063e33a603.svg?invert_in_darkmode&sanitize=true" align=middle width=15.480837899999988pt height=22.465723500000017pt/> is the set of players in our team that we have already chosen, and must be a part of our team, and <img src="/tex/c61ac6827db8079d4afe2ce171351ec9.svg?invert_in_darkmode&sanitize=true" align=middle width=18.41344229999999pt height=22.465723500000017pt/> is the set of players chosen by others that we can no longer pick, and <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is an integer representing the round of the draft we are in. Using for notational convenience, we will denote our new constraints 

<p align="center"><img src="/tex/25cb023cb4bc1463001be4d4d3e574b3.svg?invert_in_darkmode&sanitize=true" align=middle width=131.08432094999998pt height=38.54816295pt/></p> 
<p align="center"><img src="/tex/c4b64ce9ceefaac256ca562054a3055f.svg?invert_in_darkmode&sanitize=true" align=middle width=130.96137119999997pt height=38.54816295pt/></p>


so we may reference them directly in algorithm (1) below. 

**Algorithm(**_draft_**):**           (1)

---

 1. Choose a value for risk tolerance <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/>
 2.  Decide on team size **_MAX_**
 3. **_WHILE <img src="/tex/c483334a8dd61243b9a3b9d78b730f6e.svg?invert_in_darkmode&sanitize=true" align=middle width=27.92803034999999pt height=22.831056599999986pt/> MAX_ DO:**
    * Update the set <img src="/tex/4221397c8a5a02a9d784666f47094f17.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=68.74498784999999pt height=24.65753399999998pt/> of players which have been drafted
        * Update constraint (5)
    * Solve the maximization problem (3)
        * Using some metric, choose a single player <img src="/tex/a9181dcbb0c785f87c807a62fbca43d5.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=14.15517674999999pt height=14.611878600000017pt/> from the solution of (3)
    * Update the set <img src="/tex/37333fbdc33aa49b7a5c0fb483c5d522.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=66.51064694999998pt height=24.65753399999998pt/> by adding the player we have chosen to our team
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

There are many metrics by which we can choose a single player and add them to our teams set of players <img src="/tex/40c047230b846927c8e0337dadd5bd5a.svg?invert_in_darkmode&sanitize=true" align=middle width=19.84068074999999pt height=22.465723500000017pt/> for each round of the draft, and not all of them are created equal but worth exploring. In fact, many of the fantasy teams we created for this project take advantage of different metrics with respect to their choosing criteria. For example, there are the two easiest choices for choosing a single player:

1. Choose an available player <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> whose average returns <img src="/tex/3cf87ea38a615ed99e0232f8ed9431fe.svg?invert_in_darkmode&sanitize=true" align=middle width=12.067218899999991pt height=14.15524440000002pt/> are highest from the solution of equation (3). 
2. Choose an available player <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> whose RMS of returns is highest from the solution of equation (3)

These two options are the easiest to implement and worth exploring, however, with all the computational power at hand - we may also be interested in trying to optimize once again in order to find the best player to choose. For example, if we again use a Markowitz style optimization using only the players from our specific team <img src="/tex/d30de5af1ca31294a1a7debdc025329a.svg?invert_in_darkmode&sanitize=true" align=middle width=25.398989549999992pt height=22.465723500000017pt/>, where the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> subscript indicates that this is the <img src="/tex/448aa5c656d95a0430b90ac5ded7840c.svg?invert_in_darkmode&sanitize=true" align=middle width=21.737180849999987pt height=27.91243950000002pt/> round of the draft, we could write the following optimization problem

<p align="center"><img src="/tex/3eb66d3bc94fb4d0897fd412d9fec998.svg?invert_in_darkmode&sanitize=true" align=middle width=350.3477208pt height=35.76269895pt/></p>

Where in equation (6) above, the player vector <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/>, the returns vector <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> and covariance matrix <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> are limited to only the entries which are relevant to the players in the current optimal team from the solution of equation (3) in draft round <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, and <img src="/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/> is again a risk avoidance parameter. In the equation above, <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> is no longer discrete and we will return a player vector with real arguments. In this sense, the solution to the first maximization problem gives tells us how much to "invest" in a particular player for the optimal portfolio. The argument then, is if we can only pick one player, we want to pick the player with the highest investment, which is where the <img src="/tex/04fed6b232b8cef68f26f56ee7fca507.svg?invert_in_darkmode&sanitize=true" align=middle width=53.47052864999999pt height=14.15524440000002pt/> comes in. This is simply a statement that we will pick the player whose investment is greatest from the solution vector to our optimization problem. 

### Details On The Sub Optimization
Coming soon


## Binary Integer Programing
Coming soon. 

## SportsNet Fantasy Hockey

In this case, we have a slightly different problem to solve. Rather than solving several instances of an optimization problem using limited resources between players, here everyone is free to choose whatever player they want - subject to division, position, and a points value cap. Formally, 

<p align="center"><img src="/tex/afdc4d872b047c7a64bbf8e69c7b89c3.svg?invert_in_darkmode&sanitize=true" align=middle width=306.73310745pt height=212.46262124999996pt/></p>

where <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> is a binary vector of players, <img src="/tex/d303788ea8b3ff5079316016e37bf19e.svg?invert_in_darkmode&sanitize=true" align=middle width=7.785368249999991pt height=14.611878600000017pt/> is our returns vector, <img src="/tex/61ccc6d099c3b104d8de703a10b20230.svg?invert_in_darkmode&sanitize=true" align=middle width=14.20083224999999pt height=22.55708729999998pt/> is the covariance matrix, <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is the risk avoidance parameter and <img src="/tex/12d3ebda1a212bd89197298f60cf3ce1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.652895299999988pt height=22.55708729999998pt/> is a diagonal matrix of the cost, or points value assigned by SportsNet, associated with each player.

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

In this case, as playoffs are elimination based, it is not favorable for us to choose players from teams that we may expect to lose. In this case, we introduced an artificial _win bias_ into the points scoring system during the optimization by awarding each player an additional 1.5 points for each game they have won. The idea here is that our optimization will now be biased towards teams that tend to win more games, which is ideal for playoff hockey. What should be noted is that formally, we're updating our points vector for each player <img src="/tex/980fcd4213d7b5d2ffcc82ec78c27ead.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=14.611878600000017pt/> as 

<p align="center"><img src="/tex/726bdaa9a17577ca84e630d9a2869865.svg?invert_in_darkmode&sanitize=true" align=middle width=121.87418429999998pt height=14.611878599999999pt/></p>

where <img src="/tex/a10ec92d13e76a02b538967f6b90b345.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/> is a _bias_ vector with elements defined by

<p align="center"><img src="/tex/b7226844f0776ee9c35432c79840ea6d.svg?invert_in_darkmode&sanitize=true" align=middle width=110.9757825pt height=14.611878599999999pt/></p>

where <img src="/tex/027542eeba61a7e7fa1834eb3fe5f62a.svg?invert_in_darkmode&sanitize=true" align=middle width=17.12526254999999pt height=22.831056599999986pt/> is the Kroneker delta, defined as 

<p align="center"><img src="/tex/68404186ff04d90f97fb2bb6a3864665.svg?invert_in_darkmode&sanitize=true" align=middle width=118.58643224999999pt height=49.315569599999996pt/></p>

Where our calculations of our mean returns vector and covariance terms are identical, however, now we are using our win-biased points instead. 
