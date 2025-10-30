##  Prompting Guidance

> 🧠 **Note:** ChronoEdit was trained with richly captioned data. The model responds best to prompts that match the **training distribution** — detailed, structured, and visually grounded. We provide several tips to help you write effective prompts for **ChronoEdit** models.


---

### 🪶 Use the Provided Prompt Rewriter
ChronoEdit’s training data is captioned using a **VLM-based dense captioning pipeline** ([`scripts/data_captioning.py`](scripts/data_captioning.py)).  
To achieve the best results, the provided prompt rewriter would rewrite your input prompt by a LLM and ensure your inference prompt follows a similar **length and structure** to the training-time distribution — typically concise, descriptive, and naturally formatted. 
Example usage:
```bash
python scripts/prompt_enhancer.py --input-image ./assets/images/output.jpg --input-prompt "extract the person wearing red coat"
```

---

### ✍️ Be Direct — Describe the Edit Explicitly
Focus on **describing the actual visual changes**, rather than summarizing them abstractly.

**Example:**
```text
✅ Prompt: Replace the character’s outdoor clothing with a formal office outfit and change the background to a softly blurred modern office space.
❌ Prompt: Change the input image into a professional portrait.
```

The model performs better when the edit instruction mirrors the detailed caption style used during training.

---

### 🧍‍♀️ Specify Human Pose Intent
When instructing the model to edit humans, be **specific about pose changes**.  
If the pose should remain exactly the same, state it explicitly:

```text
The character’s pose should stay the same.
```

Being clear about body posture or limb positioning helps maintain consistency and realism across edits.

---

### 🎥 Define Camera Pose and Composition
Include the **camera angle, framing, and viewpoint** in your prompt to guide the visual composition.  
Useful terms include:

> *Close-Up*, *Medium Shot*, *Full Shot*, *Side View*, *Top-Down View*, *Over-the-Shoulder*, *Three-Quarter View*

For example:

```text
A medium shot of the character sitting at a desk, viewed from the side.
```

---

### 🧩 Preserve the Global Structure for Local Edits
For **local edits** such as adding or replacing an object, specify that the **overall structure of the image should remain unchanged**.  
This keeps the model’s focus limited to the edited region and prevents unintended scene modifications.

```text
The overall structure of the image remains unchanged.
```

---


### 🎯 Object Extraction or Isolation
When you want to **extract or isolate an object** (e.g., cut out a person or item), clearly indicate that the background should be removed in the first sentence.

**Example:**
```text
Extract the character from the scene while removing the background. 
```
or
```text
Keep only the car while removing the background.
```

If you intend to replace the removed background, describe the new environment explicitly:

```text
Remove the background and place the character in a bright indoor studio.
```


### 💡 General Tips
- Keep prompts **concise and visually descriptive**.  
- Avoid overlong sentences or ambiguous phrasing.  
- Clearly state **which elements change** and **which remain fixed**.  
- Use natural language with concrete nouns and visual adjectives (e.g., *“wooden chair,” “soft lighting,” “overcast sky”*).

---

