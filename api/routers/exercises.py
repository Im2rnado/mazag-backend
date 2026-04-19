"""
Exercises Router — returns curated mental health exercises.

Exercises are served from a static list (no DB needed).
Matches the Exercise type used by the Expo frontend.
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from api.schemas import ExerciseSchema

router = APIRouter(prefix="/exercises", tags=["Exercises"])

EXERCISES: List[dict] = [
    {
        "id": "e1",
        "title": "4-7-8 Breathing",
        "type": "breathing",
        "duration_minutes": 5,
        "description": "A powerful breathing technique that activates your parasympathetic nervous system to reduce anxiety and promote calm.",
        "difficulty": "beginner",
        "benefits": ["Reduces anxiety", "Lowers heart rate", "Improves sleep", "Reduces stress"],
        "steps": [
            "Find a comfortable seated position and close your eyes.",
            "Exhale completely through your mouth, making a whoosh sound.",
            "Close your mouth and inhale quietly through your nose for 4 counts.",
            "Hold your breath for 7 counts.",
            "Exhale completely through your mouth for 8 counts.",
            "Repeat the cycle 3–4 times."
        ],
    },
    {
        "id": "e2",
        "title": "Box Breathing",
        "type": "breathing",
        "duration_minutes": 4,
        "description": "A balanced breathing technique used by Navy SEALs to stay calm under pressure. Four equal counts for each phase.",
        "difficulty": "beginner",
        "benefits": ["Reduces stress", "Improves focus", "Calms the nervous system"],
        "steps": [
            "Sit upright in a comfortable position.",
            "Exhale slowly, releasing all air from your lungs.",
            "Inhale through your nose for 4 counts.",
            "Hold your breath for 4 counts.",
            "Exhale through your mouth for 4 counts.",
            "Hold for 4 counts before the next breath.",
            "Repeat 4–6 times."
        ],
    },
    {
        "id": "e3",
        "title": "Morning Pages Journaling",
        "type": "journaling",
        "duration_minutes": 15,
        "description": "Write three pages of freehand thoughts every morning. This practice clears mental clutter and boosts creativity and emotional clarity.",
        "difficulty": "beginner",
        "benefits": ["Clears mental clutter", "Boosts self-awareness", "Reduces anxiety", "Improves mood"],
        "steps": [
            "Grab a notebook and pen (handwriting works best).",
            "Immediately after waking, write continuously for 3 pages.",
            "Don't filter or re-read — just write whatever comes to mind.",
            "Cover any thought, feeling, or worry.",
            "Set a timer if helpful. Keep going even when nothing comes.",
            "After finishing, close the notebook without reading."
        ],
    },
    {
        "id": "e4",
        "title": "Gratitude Journaling",
        "type": "journaling",
        "duration_minutes": 10,
        "description": "Write three specific things you're grateful for each day. Shifts focus from problems to appreciation and builds positive neural patterns.",
        "difficulty": "beginner",
        "benefits": ["Improves mood", "Builds optimism", "Reduces negative thinking", "Better sleep"],
        "steps": [
            "Find a quiet moment (morning or evening works best).",
            "Write today's date.",
            "List 3 specific things you're genuinely grateful for today.",
            "For each item, write WHY you're grateful — one sentence.",
            "Reflect briefly on how each made you feel.",
            "Close with one positive intention for tomorrow."
        ],
    },
    {
        "id": "e5",
        "title": "Body Scan Meditation",
        "type": "meditation",
        "duration_minutes": 10,
        "description": "A mindfulness practice of systematically bringing attention to different parts of the body, releasing tension and building body awareness.",
        "difficulty": "beginner",
        "benefits": ["Releases body tension", "Reduces anxiety", "Improves sleep", "Builds mindfulness"],
        "steps": [
            "Lie down comfortably or sit with your back straight.",
            "Close your eyes and take 3 deep breaths.",
            "Bring your attention gently to the top of your head.",
            "Slowly move your attention downward — forehead, eyes, jaw, neck.",
            "Notice any tension, warmth, or sensation without judgment.",
            "Continue through shoulders, chest, stomach, arms, hands.",
            "Move to your lower back, hips, legs, and feet.",
            "If your mind wanders, gently bring it back to the body.",
            "End by taking three more deep breaths."
        ],
    },
    {
        "id": "e6",
        "title": "5-Minute Mindfulness Meditation",
        "type": "meditation",
        "duration_minutes": 5,
        "description": "A short but powerful mindfulness practice focusing on breath awareness. Perfect for daily stress management.",
        "difficulty": "beginner",
        "benefits": ["Reduces stress", "Improves focus", "Calms racing thoughts"],
        "steps": [
            "Sit comfortably, close your eyes, and set a 5-minute timer.",
            "Focus all attention on your natural breathing.",
            "Notice the sensation of air entering and leaving your nostrils.",
            "When thoughts arise (and they will), gently return to the breath.",
            "Don't try to stop thoughts — just observe them without engaging.",
            "When the timer ends, open your eyes slowly and stretch."
        ],
    },
    {
        "id": "e7",
        "title": "Progressive Muscle Relaxation",
        "type": "relaxation",
        "duration_minutes": 15,
        "description": "Systematically tense and release muscle groups throughout the body to achieve deep physical and mental relaxation.",
        "difficulty": "beginner",
        "benefits": ["Reduces muscle tension", "Lowers anxiety", "Improves sleep", "Relieves physical stress"],
        "steps": [
            "Lie down in a quiet place, close your eyes.",
            "Take three slow deep breaths.",
            "Starting with your feet: tense the muscles tightly for 5 seconds.",
            "Release and notice the relaxation for 10 seconds.",
            "Move upward: calves, thighs, stomach, hands, arms, shoulders, face.",
            "Tense each group for 5 seconds, release for 10.",
            "End with your whole body relaxed, resting in the calm."
        ],
    },
    {
        "id": "e8",
        "title": "5-4-3-2-1 Grounding",
        "type": "relaxation",
        "duration_minutes": 5,
        "description": "A sensory grounding technique to bring you back to the present moment during anxiety, panic, or overwhelm.",
        "difficulty": "beginner",
        "benefits": ["Stops panic attacks", "Reduces anxiety", "Grounds you in the present"],
        "steps": [
            "Take a slow, deep breath.",
            "Name 5 things you can SEE around you right now.",
            "Name 4 things you can physically FEEL (e.g., your clothes, the chair).",
            "Name 3 things you can HEAR in your environment.",
            "Name 2 things you can SMELL (or like to smell).",
            "Name 1 thing you can TASTE.",
            "Take another deep breath. Notice how you feel."
        ],
    },
    {
        "id": "e9",
        "title": "10-Minute Walk",
        "type": "movement",
        "duration_minutes": 10,
        "description": "A mindful walking exercise that combines light physical activity with sensory awareness to lift mood and clear the mind.",
        "difficulty": "beginner",
        "benefits": ["Boosts mood", "Reduces cortisol", "Clears thinking", "Provides gentle exercise"],
        "steps": [
            "Step outside or find a space to walk.",
            "Leave your phone behind or put it on silent.",
            "As you walk, notice your surroundings — trees, sky, sounds.",
            "Feel the ground under your feet with each step.",
            "Breathe naturally and let your mind wander freely.",
            "If worries intrude, name them ('worry', 'thought') and refocus on your surroundings.",
            "After 10 minutes, pause and take 3 deep breaths before going back."
        ],
    },
    {
        "id": "e10",
        "title": "Thought Record (CBT)",
        "type": "journaling",
        "duration_minutes": 15,
        "description": "A core CBT technique to identify and challenge negative automatic thoughts by examining evidence for and against them.",
        "difficulty": "intermediate",
        "benefits": ["Challenges negative thinking", "Reduces depression", "Builds cognitive flexibility"],
        "steps": [
            "Write down the situation that triggered your distress.",
            "Identify the automatic thought (e.g., 'I always fail').",
            "Rate your belief in that thought: 0–100%.",
            "Write ALL the evidence that SUPPORTS this thought.",
            "Write ALL the evidence that CONTRADICTS this thought.",
            "Write a more balanced, realistic alternative thought.",
            "Re-rate your belief in the original thought and your mood."
        ],
    },
    {
        "id": "e11",
        "title": "Loving-Kindness Meditation",
        "type": "meditation",
        "duration_minutes": 10,
        "description": "Cultivate compassion and love for yourself and others through this ancient meditation practice. Especially helpful for self-criticism and loneliness.",
        "difficulty": "intermediate",
        "benefits": ["Increases self-compassion", "Reduces self-criticism", "Builds empathy", "Improves relationships"],
        "steps": [
            "Sit comfortably, close your eyes, take a few deep breaths.",
            "Picture someone you love easily — perhaps a pet or dear friend.",
            "Silently repeat: 'May you be happy. May you be healthy. May you be at peace.'",
            "Feel the warmth of these wishes in your chest.",
            "Now direct these same wishes to YOURSELF: 'May I be happy. May I be healthy. May I be at peace.'",
            "Expand the circle to a neutral person, then all beings.",
            "Rest in the feeling of open-hearted connection for a few moments."
        ],
    },
    {
        "id": "e12",
        "title": "Sleep Hygiene Routine",
        "type": "relaxation",
        "duration_minutes": 30,
        "description": "A wind-down routine to prepare your mind and body for restful sleep. Targets insomnia and poor sleep quality.",
        "difficulty": "beginner",
        "benefits": ["Improves sleep quality", "Reduces insomnia", "Lowers night anxiety", "Promotes deep rest"],
        "steps": [
            "60 minutes before bed: dim all lights in your home.",
            "Put all screens away or use night mode and blue-light glasses.",
            "Do a light 5-minute stretch or progressive muscle relaxation.",
            "Write 3 things you're grateful for in a journal.",
            "Set tomorrow's top 3 priorities so your brain can let them go.",
            "Keep your bedroom cool (18–20°C) and dark.",
            "Practice 4-7-8 breathing for 5 minutes in bed.",
            "If you can't sleep after 20 minutes, get up and read until sleepy."
        ],
    },
]


@router.get("", response_model=List[ExerciseSchema])
async def get_exercises(
    type: Optional[str] = Query(None, description="Filter by type: breathing, journaling, meditation, relaxation, movement"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty: beginner, intermediate, advanced"),
):
    """Return all exercises, optionally filtered by type and difficulty."""
    results = EXERCISES[:]

    if type:
        results = [e for e in results if e["type"] == type.lower()]
    if difficulty:
        results = [e for e in results if e.get("difficulty", "") == difficulty.lower()]

    return [
        ExerciseSchema(
            id=e["id"],
            title=e["title"],
            type=e["type"],
            duration_minutes=e.get("duration_minutes"),
            description=e.get("description"),
            difficulty=e.get("difficulty"),
            benefits=e.get("benefits", []),
            steps=e.get("steps"),
        )
        for e in results
    ]
