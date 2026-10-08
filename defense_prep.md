# Defense Prep: Explained Simply

How to use this file:
- Every topic has 3 parts:
  - 🟢 **Simple idea**: what it means, in plain words
  - 🔢 **Example**: with numbers from YOUR project
  - 🎤 **Say this**: a short answer you can say in the defense
- Read it top to bottom once. Then practice the 🎤 parts out loud.

---

## PART 0: What did my project do? (Learn this first!)

🟢 **Simple idea**

I have 27,000 **satellite photos** of Europe. Each photo shows one type of land: forest, river, highway, farm field, city houses, sea, and so on (10 types in total).

**My goal:** teach a computer to look at a photo and say which of the 10 types it is.

I tried **3 different "brains"** (models) and compared them:

| Model | What it is (simple) | Correct answers on new photos |
|---|---|---|
| **MLP** | simple network that looks at pixels one by one | **61.6%** 😐 |
| **CNN** | network made for images that looks at small areas | **90.3%** 🙂 |
| **ResNet18** | big network that already learned to "see" from 1 million photos | **97.1%** 😃 |

🎤 **Say this:**
> "I classified satellite images into 10 land types. A simple MLP got 61.6%, my own CNN got 90.3%, and a pretrained ResNet18 got 97.1%. I also compared 4 optimizers and tested dropout against overfitting."

---

## PART 1: The data

### 1.1 How did I split the data?

🟢 **Simple idea:** it works like school.
- **Training set (70% = 18,900 photos)** = the **textbook**. The model studies from these.
- **Validation set (15% = 4,050 photos)** = **practice tests**. I check progress during studying.
- **Test set (15% = 4,050 photos)** = the **final exam**. Used only **once**, at the very end.

Why? If the model saw the exam questions while studying, its score would be fake.

🎤 **Say this:**
> "I split the data 70/15/15 with a fixed random seed. Training data is for learning, validation is for checking during training, and test is used only once at the end for an honest score."

### 1.2 Normalization

🟢 **Simple idea:** pixel values become small numbers centred around 0. All 3 colour channels (Red, Green, Blue) then have a similar range, and the network learns faster and more smoothly.

I computed the average and spread **only from the training photos**, because using the test photos would be "peeking at the exam".

🎤 **Say this:**
> "I normalized each colour channel using mean and standard deviation from the training set only, to avoid data leakage."

### 1.3 Data augmentation (flips)

🟢 **Simple idea:** during training I randomly **flip photos** left-right and up-down.
- A satellite photo has no "up". A flipped forest is still a forest.
- So the model sees "new" photos for free and memorizes less.
- I do this **only for training**, never for validation or test.

### 1.4 Class balance

🔢 Most classes have 3,000 photos, some have 2,500, and Pasture has 2,000. That's **almost balanced**, so "accuracy" is a fair way to measure.

---

## PART 2: The MLP (the simple model)

### 2.1 How does an MLP work?

🟢 **Simple idea:**
1. Take the 64×64 colour photo.
2. **Flatten** it: put all the pixel numbers in **one long line**. 64 × 64 × 3 colours = **12,288 numbers**.
3. Each neuron in the next layer looks at **all 12,288 numbers**, multiplies each by a weight, adds them up, then applies ReLU.

My MLP:
```
12,288 numbers  →  512 neurons  →  256 neurons  →  10 outputs (one per class)
```
The output with the highest score is the prediction.

### 2.2 Why is the MLP bad for images? ⭐ (they WILL ask this)

🟢 **Simple idea:** flattening is like **cutting a photo into pixels and throwing them into a line**. The MLP **doesn't know which pixels are next to each other**.
- A river is a shape made of **neighbouring** pixels, but the MLP can't see shapes.
- So it mostly uses **colour**. That's why it confuses **dark-green forest with dark sea** (78 mistakes!).
- It also needs a lot of weights: the first layer alone has 12,288 × 512 ≈ **6.3 million** weights.

🎤 **Say this:**
> "The MLP flattens the image, so it loses which pixels are neighbours. It can't detect shapes, only colours, so it got only 61.6%. It also has 6.4 million parameters, three times more than my CNN."

### 2.3 What is ReLU?

🟢 **Simple idea:** `ReLU(x) = max(0, x)` → **negative numbers become 0, positive numbers stay the same.**
- Without it, the whole network would be just one big linear formula and couldn't learn complex things.
- It's simple and fast, and it doesn't cause the "vanishing gradient" problem (explained in Part 5).

---

## PART 3: The CNN (my image model) ⭐⭐ MOST IMPORTANT

### 3.1 What is a convolution?

🟢 **Simple idea:** imagine a **small 3×3 window (a "filter")** that you **slide over the photo**, step by step, from the top-left to the bottom-right.
- At every position it looks at only **9 neighbouring pixels** and calculates **one number**: "how much does this spot look like my pattern?"
- One filter might look for **vertical edges**, another for **green colour**, and so on.
- The result of sliding one filter over the whole photo is a new image called a **feature map**, a "heat map" of where that pattern appears.

```
 Photo (64×64)                 Feature map (64×64)
┌──────────────┐  slide 3×3   ┌──────────────┐
│ ▓▓░░░░░░     │  filter  →   │ bright where │
│ ▓▓░░░░░░     │              │ the pattern  │
│ ...          │              │ was found    │
└──────────────┘              └──────────────┘
```

**Why is this better than an MLP?**
- It looks at **neighbouring pixels together**, so it can see **edges and shapes**.
- The **same filter is reused everywhere** ("weight sharing"). A forest pattern is recognized wherever it appears, and you need far fewer weights.

### 3.2 Kernel size, stride, padding

🟢 **Simple idea:**
- **Kernel size (K)** = size of the window. I use **3×3**.
- **Stride (S)** = **how many pixels the window jumps** each step. I use **1**, so it moves 1 pixel at a time and nothing is skipped. (Stride 2 jumps 2 pixels, so the output becomes half the size.)
- **Padding (P)** = a **frame of zeros around the photo**. I use **1** (a 1-pixel frame). Without it, the window can't be centred on border pixels, so the output gets smaller. With padding 1 and a 3×3 kernel, the **output size = input size**.
- **Number of filters** = how many different patterns to look for = how many feature maps come out. I use **32, then 64, then 128**.

### 3.3 Output size formula ⭐⭐⭐ (they WILL ask you to calculate this)

$$\text{Output} = \frac{W - K + 2P}{S} + 1$$

| Letter | Meaning | My Conv1 |
|---|---|---|
| W | input width | 64 |
| K | kernel size | 3 |
| P | padding | 1 |
| S | stride | 1 |

**Step by step for my Conv1:**
1. W − K = 64 − 3 = **61**
2. 2P = 2 × 1 = **2** → 61 + 2 = **63**
3. Divide by S: 63 / 1 = **63**
4. Add 1: 63 + 1 = **64** ✅

So the output is **64×64**, with **32 channels** (because 32 filters) → **32 × 64 × 64**.

**Then MaxPool (window 2, stride 2):** (64 − 2 + 0) / 2 + 1 = 31 + 1 = **32** → **32 × 32 × 32**

**Practice: cover the answer and try yourself!**

| W | K | P | S | Calculation | Answer |
|---|---|---|---|---|---|
| 64 | 3 | 0 | 1 | (64−3+0)/1 + 1 | **62** (no padding → shrinks) |
| 64 | 5 | 2 | 1 | (64−5+4)/1 + 1 | **64** |
| 64 | 3 | 1 | 2 | (64−3+2)/2 + 1 = 31.5 → round **down** to 31, +1 | **32** |
| 32 | 3 | 1 | 1 | (32−3+2)/1 + 1 | **32** |

💡 If the division is not a whole number, **round down**.

### 3.4 Pooling

🟢 **Simple idea:** **Max pooling 2×2** looks at every 2×2 block of 4 numbers and **keeps only the biggest one**.
- The image becomes **half as wide and half as tall** (64 → 32).
- It keeps the strongest signal ("the pattern was found here").
- It makes the network faster and less sensitive to small shifts.
- It has **no weights to learn** (0 parameters).

```
 1  3 | 2  0
 5  2 | 1  4        →     5  4
------+------              7  9
 7  1 | 9  3
 0  2 | 6  8
```

### 3.5 My whole CNN, step by step

```
Photo                  3 × 64 × 64     (3 colours)
Conv 32 filters + ReLU → 32 × 64 × 64
MaxPool                → 32 × 32 × 32  (half size)
Conv 64 filters + ReLU → 64 × 32 × 32
MaxPool                → 64 × 16 × 16
Conv 128 filters + ReLU→ 128 × 16 × 16
MaxPool                → 128 × 8 × 8
Flatten                → 8,192 numbers (128×8×8)
Dense layer            → 256
Dropout 0.5
Dense layer            → 10 outputs (one per class)
```

💡 **Pattern to remember:** the image gets **smaller** (64 → 32 → 16 → 8), and the number of filters gets **bigger** (32 → 64 → 128). Small details are combined into bigger, more meaningful patterns.

### 3.6 Counting parameters (weights) ⭐⭐

🟢 **Simple idea:** a **parameter** = one number the network learns.

**For a conv layer:** each filter has (3 × 3 × input channels) weights + 1 bias.
Total = (3 × 3 × input channels + 1) × number of filters

| Layer | Calculation | Parameters |
|---|---|---|
| Conv1 | (3·3·**3** + 1) × 32 = 28 × 32 | **896** |
| Conv2 | (3·3·**32** + 1) × 64 = 289 × 64 | **18,496** |
| Conv3 | (3·3·**64** + 1) × 128 = 577 × 128 | **73,856** |
| Dense1 | 8192 × 256 + 256 | **2,097,408** |
| Dense2 | 256 × 10 + 10 | **2,570** |
| **Total** | | **2,193,226** |

💡 Why "3" in Conv1? The photo has **3 colour channels** (RGB). In Conv2 it's 32, because Conv1 produced 32 feature maps.
💡 ReLU, MaxPool and Dropout have **0 parameters**. They only do a fixed operation.

🎤 **Say this:**
> "My CNN has 3 conv blocks with 3×3 kernels, stride 1 and padding 1, so convolution keeps the size, and max pooling halves it each time: 64 → 32 → 16 → 8. Total parameters are 2.19 million, fewer than the MLP, but it's much more accurate because it uses neighbouring pixels and shares weights."

### 3.7 What did the first layer learn? (feature visualization)

🟢 **Simple idea:** I showed the first-layer filters and their **feature maps** for two photos.
- Some maps light up on **edges**: field borders, paths, river banks.
- Some light up on **bright or coloured areas**.
- Some show fine **texture**.
- **The first layers find simple things (edges, colours).** Deeper layers combine them into bigger things (field stripes, street patterns, river shapes).

---

## PART 4: How training works (basics)

🟢 **Simple idea:** **training = practice + correction, repeated many times.**

1. Show the network **128 photos** (one **batch**).
2. It makes guesses.
3. **Loss** = a number that says **how wrong** the guesses were. I use **Cross-Entropy loss**, the standard one for choosing between classes.
4. **Backpropagation** works out, for each weight, "should it go up or down to reduce the loss?" This is the **gradient**.
5. The **optimizer** changes the weights a little in that direction.
6. Repeat. **One epoch** = all 18,900 training photos seen once ≈ **148 batches**.

🔢 **The code does exactly this:**
```python
optimizer.zero_grad()   # clear old gradients
out = model(x)          # 1-2. make guesses
loss = criterion(out, y)# 3. how wrong?
loss.backward()         # 4. compute gradients (backpropagation)
optimizer.step()        # 5. update the weights
```

🟢 **Comparison:** you're **blindfolded on a mountain** and want to reach the **lowest valley** (lowest loss).
- The **gradient** tells you which way is downhill.
- The **learning rate** is **how big a step** you take.
  - Too big → you jump over the valley.
  - Too small → it takes forever.

---

## PART 5: Optimizers ⭐⭐

I trained the **same CNN 4 times**. Same photos, same starting weights, 5 epochs each. **Only the optimizer was different.**

| Optimizer | Simple idea | Result (validation accuracy) |
|---|---|---|
| **SGD** | Take a small fixed step downhill. | 58.9% 🐢 slow |
| **SGD + Momentum** | Like a **rolling ball**: it keeps speed from earlier steps, so it moves faster. | 82.0% |
| **RMSprop** | **Every weight gets its own step size.** Weights that change a lot take smaller steps, weights that barely move take bigger steps. | 83.1% |
| **Adam** | **Momentum + RMSprop together.** The rolling ball, plus an individual step size per weight. | **85.7%** 🏆 |

🔢 **Time:** all 4 took ≈ **70 seconds**. The optimizer doesn't make each step slower. It changes **how many steps you need to reach a good result**.

🎤 **Say this:**
> "Plain SGD was the slowest learner, only 58.9% after 5 epochs. Momentum adds speed from previous steps, RMSprop gives each weight its own step size, and Adam combines both. Adam was best with 85.7%. All four took the same time per epoch. The difference is how fast they converge."

💡 **"Converge"** = reach a good, stable result.

---

## PART 6: Overfitting and Dropout ⭐⭐

### 6.1 What is overfitting?

🟢 **Simple idea:** a student who **memorizes the answers** in the textbook instead of **understanding**.
- They get 100% on textbook questions, but fail on **new** exam questions.
- For a network: **training accuracy is very high, but validation accuracy is lower**, and the gap between them grows.

**How to see it in the plots:** the training loss keeps going **down**, but the validation loss starts going **up**.

### 6.2 What is dropout?

🟢 **Simple idea:** during training, **randomly switch off 50% of the neurons** each step. Different ones every time.
- **Comparison:** a football team where random players are missing in each practice. **Everyone must learn to play well**, so the team doesn't depend on one star player.
- The network can't rely on a few neurons, so it learns **more general, robust patterns**.
- During testing **all neurons are ON** (`model.eval()` does this).

### 6.3 My experiment

| | Training acc | Validation acc | Gap |
|---|---|---|---|
| **No dropout** | 95.8% | 90.7% | **5.1%** ← memorizing |
| **Dropout 0.5** | 93.3% | **91.4%** | **1.9%** ← generalizing |

- **No dropout:** high training score, but the validation loss started going **up** after epoch 8, which means overfitting.
- **With dropout:** training score is a bit **lower** (training is harder on purpose), but **validation is higher** and the gap is much smaller.

🎤 **Say this:**
> "Without dropout the model started overfitting: the train-validation gap was 5.1% and validation loss rose at the end. With dropout 0.5 the gap dropped to 1.9% and validation accuracy improved slightly, to 91.4%. Dropout forces the network to learn robust features instead of memorizing."

💡 I turned off flips in this experiment, because flips also reduce overfitting and would hide the dropout effect.

### 6.4 Other words they might ask about

- **Underfitting:** the model is too weak, so even the training score is low. Example: my MLP at about 60%.
- **Vanishing gradient:** with the **sigmoid** activation, the correction signal gets **multiplied by small numbers** (at most 0.25) in every layer. In a deep network it becomes almost **zero**, so the first layers stop learning. **ReLU** fixes this, because its gradient is 1 for positive numbers.
- **Weight decay (L2):** a penalty for big weights, which keeps the model simple.
- **Early stopping:** stop training when the validation loss stops improving. In my no-dropout run, that would be epoch 8.

---

## PART 7: Transfer learning (ResNet18) ⭐⭐

### 7.1 Simple idea

🟢 **Comparison:** teaching someone to recognize land types from satellite photos.
- **Option A:** a **newborn baby**, who must first learn what edges, colours and shapes even are. → This is my CNN, starting from zero.
- **Option B:** an **adult** who has already seen millions of pictures. They already understand edges, textures and shapes, so they just need to learn the 10 new categories. → This is **transfer learning**.

**ResNet18** was already trained on **ImageNet**: 1.2 million photos of 1,000 everyday things (dogs, cars, and so on). Its "eyes" (early layers) already know edges, textures and shapes, and those are useful for satellite photos too.

### 7.2 What I did (3 steps)

1. **Loaded ResNet18 with its ImageNet knowledge** (pretrained weights).
2. **Replaced the last layer.** Originally it outputs 1,000 classes. I replaced it with a new layer that outputs **10 classes** (my land types).
3. **Fine-tuned:** trained the whole network for **only 3 epochs** with a **small learning rate (0.0001)**. A small rate makes only **small adjustments**, so its existing knowledge isn't destroyed.

### 7.3 Result

🔢 **97.1%**, the best of all models. After just **1 epoch** it already had 95.6% validation accuracy.

### 7.4 What is the "Res" in ResNet?

🟢 **Simple idea:** ResNet has **shortcuts** (skip connections) that let the input **jump over** a few layers and be added back: `output = layers(x) + x`.
- The learning signal can travel back through these shortcuts **without fading**, so very deep networks can be trained (no vanishing gradient).

🎤 **Say this:**
> "Transfer learning means reusing a model that already learned general visual features on ImageNet. I loaded pretrained ResNet18, replaced the last layer from 1,000 to 10 classes, and fine-tuned with a small learning rate for 3 epochs. It reached 97.1%, the best result, because it didn't have to learn edges and textures from zero."

---

## PART 8: Comparing the results ⭐

### 8.1 Big comparison table

| | MLP | CNN | ResNet18 |
|---|---|---|---|
| Test accuracy | 61.6% | 90.3% | **97.1%** |
| Parameters | 6.4 M | **2.2 M** (smallest) | 11.2 M (biggest) |
| Epochs | 10 | 10 | **3** |
| Total training time | ~77 s | ~143 s | ~74 s |
| Time per epoch | ~8 s | ~14 s | ~25 s |

**3 lessons from this table:**
1. **More parameters ≠ better.** The MLP has 3× more parameters than the CNN, but is much worse. The **type** of model matters more than its size.
2. **The CNN is slower per epoch than the MLP, even though it's smaller.** Its filters slide over **every position** in the image, and that means many calculations.
3. **ResNet18 is the slowest per epoch, but needed only 3 epochs.** Total time was the lowest, and accuracy was the best. Its downside: it's a big model, slower to use on small devices.

### 8.2 Confusion matrix

🟢 **Simple idea:** a table showing **what the true class was (rows)** vs **what the model guessed (columns)**.
- The **diagonal** (top-left to bottom-right) = **correct answers**.
- Everything else = **mistakes**, showing which classes get mixed up.

🔢 **Interesting mistakes:**
- **MLP:** SeaLake → Forest (78 times). Both are **dark**, and the MLP only sees colour.
- **CNN:** Highway ↔ River. Both are **long thin lines** across the image.
- **ResNet18:** almost no mistakes. River → Highway (14), Industrial → Residential (10). These are hard even for people.

### 8.3 Limitations and improvements (they often ask this at the end)

**Limitations (weak points):**
- Short training (10 epochs, and 3 for ResNet).
- Ran only once (one random seed).
- Used only RGB colour. EuroSAT also has infrared and other bands.
- All photos are from Europe.

**Improvements:**
- Train longer and lower the learning rate over time.
- Add early stopping and Batch Normalization.
- Use more augmentation (rotations, colour changes).
- Try bigger models, or use all 13 satellite bands.

---

## PART 9: Glossary (quick meanings)

| Word | Simple meaning |
|---|---|
| **Neuron** | multiplies inputs by weights, adds them up, applies an activation |
| **Weight / parameter** | a number the network learns |
| **Activation (ReLU)** | makes the network non-linear; ReLU turns negatives into 0 |
| **Epoch** | the model has seen all training photos once |
| **Batch** | a small group of photos (128) processed together |
| **Loss** | how wrong the model is (lower = better) |
| **Gradient** | which direction to change each weight to reduce the loss |
| **Backpropagation** | computes gradients from the output back to the input |
| **Learning rate** | step size when changing weights |
| **Optimizer** | the rule for changing weights (SGD, Adam, ...) |
| **Filter / kernel** | small window (3×3) that slides over the image looking for a pattern |
| **Feature map** | output of one filter: where its pattern was found |
| **Stride** | how many pixels the filter jumps |
| **Padding** | frame of zeros around the image |
| **Pooling** | shrinks the image by keeping the max of each 2×2 block |
| **Overfitting** | memorizing training data, bad on new data |
| **Dropout** | randomly turns off neurons during training to stop memorizing |
| **Transfer learning** | reusing a model trained on another big dataset |
| **Fine-tuning** | training that reused model a little more on your data |
| **Softmax** | turns output scores into probabilities (inside the loss function) |
| **Seed** | fixed starting point for randomness, so results are repeatable |

---

## PART 10: Night-before checklist ✅

- [ ] Say the Part 0 summary out loud without reading
- [ ] Calculate the output size for: W=64, K=3, P=1, S=1 (answer: 64) and W=64, K=3, P=0, S=1 (answer: 62)
- [ ] Calculate Conv1 parameters: (3·3·3+1)·32 = 896
- [ ] Explain: why is the CNN better than the MLP? (neighbouring pixels + weight sharing)
- [ ] Explain: SGD vs Momentum vs Adam (small step / rolling ball / own step size + rolling ball)
- [ ] Explain overfitting and dropout with your numbers (gap 5.1% → 1.9%)
- [ ] Explain transfer learning with the "adult vs baby" comparison
- [ ] Open the notebook and scroll through it once
