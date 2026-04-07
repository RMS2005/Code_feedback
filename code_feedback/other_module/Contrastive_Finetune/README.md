Methodology: Multi-Label Supervised Contrastive Finetuning
To create embeddings that are aware of both the problem type (e.g., Fibonacci, Palindrome) and the correctness of the solution (e.g., PASS, PARTIAL, FAIL), we fine-tune a base code embedding model using a Multi-Label Supervised Contrastive Loss (MulSupCon), inspired by the work of Zhang et al. (AAAI 2024).
This approach teaches the model to group similar solutions in the embedding space based on shared characteristics. For instance, two correct solutions to the "Fibonacci" problem should be semantically closer to each other than a correct solution is to a buggy one.
Mathematical Formulation
Let 
B
=
{
(
z
i
,
y
i
)
}
i
=
1
N
B={(z 
i
​
 ,y 
i
​
 )} 
i=1
N
​
 
 be a batch of 
N
N
 samples, where 
z
i
∈
R
D
z 
i
​
 ∈R 
D
 
 is the D-dimensional embedding of a code snippet and 
y
i
∈
{
0
,
1
}
C
y 
i
​
 ∈{0,1} 
C
 
 is its corresponding multi-hot label vector over 
C
C
 total classes. For our task, the classes include problem types (e.g., problem_q6) and performance tiers (e.g., tier_PASS).
The core idea is to maximize the similarity between an "anchor" sample 
z
i
z 
i
​
 
 and its "positive" samples (other samples in the batch that share at least one label with it), while minimizing its similarity to "negative" samples (all other samples).
The loss for a single anchor sample 
i
i
 is defined based on the log-probabilities derived from a temperature-scaled similarity matrix.
1. Similarity and Logits:
First, we compute the cosine similarity between all pairs of normalized embeddings in the batch. Let 
z
i
′
=
z
i
∥
z
i
∥
2
z 
i
′
​
 = 
∥z 
i
​
 ∥ 
2
​
 
z 
i
​
 
​
 
. The similarity matrix 
S
S
 is given by 
S
i
j
=
z
i
′
⋅
z
j
′
S 
ij
​
 =z 
i
′
​
 ⋅z 
j
′
​
 
.
These similarities are scaled by a temperature parameter 
τ
τ
 to create logits:
logits
i
j
=
z
i
′
⋅
z
j
′
τ
logits 
ij
​
 = 
τ
z 
i
′
​
 ⋅z 
j
′
​
 
​
 
2. Per-Class Positive Sets:
For the multi-label case, we define a set of positive samples for an anchor 
i
i
 with respect to a specific class k. The set of indices of positive samples for anchor 
i
i
 and class 
k
k
, denoted 
P
(
i
,
k
)
P(i,k)
, includes all other samples 
j
j
 in the batch that also possess class 
k
k
:
P
(
i
,
k
)
=
{
j
∈
{
1
,
.
.
.
,
N
}
∖
{
i
}
∣
y
i
k
=
1
 and 
y
j
k
=
1
}
P(i,k)={j∈{1,...,N}∖{i}∣y 
ik
​
 =1 and y 
jk
​
 =1}
where 
y
i
k
y 
ik
​
 
 is the k-th component of the label vector for sample 
i
i
.
3. The MulSupCon Loss Formula:
The loss for a single anchor 
i
i
 is the sum of the contrastive losses calculated for each of its labels. The loss for anchor 
i
i
 with respect to one of its classes 
k
k
 (where 
y
i
k
=
1
y 
ik
​
 =1
) is:
L
i
,
k
=
−
1
∣
P
(
i
,
k
)
∣
∑
j
∈
P
(
i
,
k
)
(
logits
i
j
−
log
⁡
∑
m
∈
{
1
,
.
.
.
,
N
}
∖
{
i
}
exp
⁡
(
logits
i
m
)
)
L 
i,k
​
 =− 
∣P(i,k)∣
1
​
  
j∈P(i,k)
∑
​
  
​
 logits 
ij
​
 −log 
m∈{1,...,N}∖{i}
∑
​
 exp(logits 
im
​
 ) 
​
 
This formula represents the negative log-likelihood of correctly identifying the positive samples for class 
k
k
 among all other samples in the batch. The term inside the sum is the log-probability of sample 
j
j
 being a positive for anchor 
i
i
.
The total loss for anchor 
i
i
 is the sum of these per-class losses over all classes it belongs to:
L
i
=
∑
k
=
1
C
y
i
k
⋅
L
i
,
k
L 
i
​
 = 
k=1
∑
C
​
 y 
ik
​
 ⋅L 
i,k
​
 
Finally, the total Multi-Label Supervised Contrastive Loss for the entire batch is the average of the losses for each anchor:
L
MulSupCon
=
1
N
∑
i
=
1
N
L
i
L 
MulSupCon
​
 = 
N
1
​
  
i=1
∑
N
​
 L 
i
​
 
This formulation effectively trains the embedding space to be structured along multiple semantic axes simultaneously, creating rich, queryable representations of student code that reflect both algorithmic strategy and correctness.


Of course. Here is the mathematical formulation of the **Multi-Label Supervised Contrastive Loss (MulSupCon)** as implemented in your `supcon_loss.py` file. This is formatted in Markdown, ready for your `README.md` or research paper.

The explanation breaks down the formula step-by-step, mirroring the logic in your Python code, making it clear and easy to follow.

---

### Mathematical Formulation of the Multi-Label Supervised Contrastive Loss

The goal of the Multi-Label Supervised Contrastive Loss (MulSupCon) is to structure an embedding space such that samples sharing one or more labels are pulled closer together, while samples with no shared labels are pushed apart. The loss is calculated independently for each label and then aggregated. This formulation is based on the methodology presented by Zhang et al. in "Multi-Label Supervised Contrastive Learning" (AAAI 2024).

Let $\mathcal{B} = \{ (\mathbf{z}_i, \mathbf{y}_i) \}_{i=1}^{N}$ be a batch of $N$ samples from the dataset.
- $\mathbf{z}_i \in \mathbb{R}^D$ is the D-dimensional embedding for the $i$-th code snippet, L2-normalized such that $\|\mathbf{z}_i\|_2 = 1$.
- $\mathbf{y}_i \in \{0, 1\}^C$ is the multi-hot label vector for the $i$-th snippet over $C$ total classes. For our task, these classes represent both the problem ID (e.g., `problem_q7`) and the performance tier (e.g., `tier_PASS`). $\mathbf{y}_{ik} = 1$ if sample $i$ has class $k$, and 0 otherwise.

---

#### **Step 1: Similarity and Log-Probabilities**

First, we compute the pairwise cosine similarity between all normalized embeddings in the batch. This forms a similarity matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$, where $\mathbf{S}_{ij} = \mathbf{z}_i \cdot \mathbf{z}_j$.

These similarities are then scaled by a temperature parameter $\tau > 0$ to produce logits, which control the sharpness of the probability distribution.

$$
\text{logits}_{ij} = \frac{\mathbf{S}_{ij}}{\tau} = \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\tau}
$$

The log-probability of correctly identifying a sample $j$ as a positive for an anchor sample $i$ (among all other samples $m \neq i$ in the batch) is given by the log-softmax function:

$$
\log P_{ij} = \text{logits}_{ij} - \log \sum_{m=1, m \neq i}^{N} \exp(\text{logits}_{im})
$$

---

#### **Step 2: Per-Class Loss Calculation**

This is the core of the multi-label approach. Instead of a single set of positives for each anchor, we define positive sets for each of the $C$ classes.

For an anchor sample $i$ and a specific class $k$, the set of indices of its positive samples, $P(i, k)$, includes all other samples $j$ in the batch that also possess class $k$.

$$
P(i, k) = \{ j \in \{1, \dots, N\} \setminus \{i\} \mid \mathbf{y}_{ik} = 1 \text{ and } \mathbf{y}_{jk} = 1 \}
$$

The supervised contrastive loss for anchor $i$ *with respect to class k* is the average of the negative log-probabilities over all its positive samples for that class. This loss is only calculated if the positive set $P(i, k)$ is not empty ($|P(i, k)| > 0$).

$$
\mathcal{L}_{i,k} = \begin{cases}
-\frac{1}{|P(i, k)|} \sum_{j \in P(i, k)} \log P_{ij} & \text{if } |P(i, k)| > 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

#### **Step 3: Aggregating the Final Loss**

The total loss for a single anchor sample $i$ is the sum of its per-class losses, calculated only for the classes it actually possesses. The multi-hot label vector $\mathbf{y}_i$ acts as a mask for this summation.

$$
\mathcal{L}_i = \sum_{k=1}^{C} \mathbf{y}_{ik} \cdot \mathcal{L}_{i,k}
$$

Finally, the total Multi-Label Supervised Contrastive Loss for the entire batch $\mathcal{B}$ is the mean of the individual anchor losses. We only average over anchors that have at least one label to avoid division by zero if a sample has no labels. Let $\mathcal{I}^+ = \{i \mid \sum_{k=1}^C \mathbf{y}_{ik} > 0\}$ be the set of indices of anchors with at least one label.

$$
\mathcal{L}_{\text{MulSupCon}} = \frac{1}{|\mathcal{I}^+|} \sum_{i \in \mathcal{I}^+} \mathcal{L}_i
$$

This formulation trains the embedding space to be structured along multiple semantic axes simultaneously. It encourages the model to learn representations that cluster code based on both the problem being solved and the quality of the solution, creating rich, queryable embeddings ideal for educational analytics.