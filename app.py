import os
import uvicorn
import requests

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from ddgs import DDGS


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set."
    )


# ============================================================
# 2. GEMMA AGENT CORE
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. JOB OPPORTUNITY TOOL
# ============================================================

@tool
def job_search(target_role: str) -> str:
    """
    Search for job and internship opportunities
    related to the student's target role.
    """

    query = (
        f"{target_role} jobs internships India "
        f"freshers campus placement"
    )

    try:

        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No job opportunities were found."

        output = []

        for item in results:

            title = item.get(
                "title",
                "Job Opportunity"
            )

            url = item.get(
                "href",
                ""
            )

            description = item.get(
                "body",
                ""
            )

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return (
            "Job search failed: "
            + str(e)
        )


# ============================================================
# 4. SKILL GAP TOOL
# ============================================================

@tool
def skill_gap(
    resume_text: str,
    target_role: str
) -> str:
    """
    Compare the student's resume with the
    requirements of the target role.
    """

    prompt = f"""

You are a campus placement skill-gap evaluator.

TARGET ROLE:
{target_role}

STUDENT RESUME:
{resume_text}

Analyze the resume against the target role.

Identify:

1. Current skills
2. Important skills required for the role
3. Skill gaps
4. High-priority gaps
5. Medium-priority gaps
6. What the student should learn next

IMPORTANT:

Only identify a current skill if it is supported
by the resume.

Do not invent skills.

Return clear normal text.
Do not return JSON.
"""

    try:

        response = llm_flash.invoke(
            prompt
        )

        content = response.content

        if isinstance(
            content,
            str
        ):

            return content.strip()

        return str(content)

    except Exception as e:

        return (
            "Skill-gap analysis failed: "
            + str(e)
        )


# ============================================================
# 5. PROJECT RECOMMENDATION TOOL
# ============================================================

@tool
def project_recommendation(
    target_role: str,
    skill_gaps: str
) -> str:
    """
    Recommend portfolio projects that address
    the student's skill gaps.
    """

    prompt = f"""

You are a technical mentor helping a student
prepare for campus placements.

TARGET ROLE:
{target_role}

SKILL GAP ANALYSIS:
{skill_gaps}

Recommend 3 practical portfolio projects.

For each project provide:

Project Name:
Technology Stack:
Objective:
Skills Improved:
Why It Helps for Placement:

Projects must directly address the identified
skill gaps.

Return normal readable text.
Do not return JSON.
"""

    try:

        response = llm_flash.invoke(
            prompt
        )

        content = response.content

        if isinstance(
            content,
            str
        ):

            return content.strip()

        return str(content)

    except Exception as e:

        return (
            "Project recommendation failed: "
            + str(e)
        )


# ============================================================
# 6. GITHUB EVALUATION TOOL
# ============================================================

@tool
def github_check(
    github_id: str
) -> str:
    """
    Evaluate the student's public GitHub profile
    and repositories.
    """

    headers = {
        "Accept":
            "application/vnd.github+json",

        "User-Agent":
            "Placement-Ready-AI-Agent"
    }

    user_url = (
        f"https://api.github.com/users/{github_id}"
    )

    repos_url = (
        f"https://api.github.com/users/"
        f"{github_id}/repos"
        f"?per_page=10&sort=updated"
    )

    try:

        user_response = requests.get(
            user_url,
            headers=headers,
            timeout=15
        )

        if user_response.status_code != 200:

            return (
                f"GitHub profile '{github_id}' "
                "could not be accessed publicly."
            )

        user = user_response.json()


        repos_response = requests.get(
            repos_url,
            headers=headers,
            timeout=15
        )

        if repos_response.status_code == 200:

            repos = repos_response.json()

        else:

            repos = []


        output = []

        output.append(
            f"Username: "
            f"{user.get('login', 'N/A')}"
        )

        output.append(
            f"Profile: "
            f"{user.get('html_url', 'N/A')}"
        )

        output.append(
            f"Public Repositories: "
            f"{user.get('public_repos', 0)}"
        )

        output.append(
            f"Followers: "
            f"{user.get('followers', 0)}"
        )

        output.append(
            "\nRecent Repositories:"
        )


        if repos:

            for repo in repos:

                output.append(

                    f"\nRepository: "
                    f"{repo.get('name', 'N/A')}\n"

                    f"Language: "
                    f"{repo.get('language', 'N/A')}\n"

                    f"Stars: "
                    f"{repo.get('stargazers_count', 0)}\n"

                    f"URL: "
                    f"{repo.get('html_url', 'N/A')}\n"

                    f"Description: "
                    f"{repo.get('description') or 'No description'}"
                )

        else:

            output.append(
                "No public repositories found."
            )


        return "\n".join(output)

    except Exception as e:

        return (
            "GitHub evaluation failed: "
            + str(e)
        )


# ============================================================
# 7. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    job_search,
    skill_gap,
    project_recommendation,
    github_check
]


# ============================================================
# 8. PLACEMENT-READY AGENT
# ============================================================

system_prompt = """

You are a Placement-Ready AI Agent.

Your purpose is to help students prepare for
campus placements.

The student provides:

1. Resume
2. Target Role
3. GitHub ID


YOUR WORKFLOW:

STEP 1:
Analyze the target role and use the Job Search
tool to find relevant job opportunities.

STEP 2:
Use the Skill Gap tool to compare the student's
resume with the target role.

STEP 3:
Use the Project Recommendation tool to recommend
projects based on the identified skill gaps.

STEP 4:
Use the GitHub Check tool to evaluate the student's
public GitHub profile and repositories.

STEP 5:
Combine the results and produce the final
Placement-Ready Report.


IMPORTANT:

YOU ARE THE AGENT CORE.

You must decide when a tool is required.

Use the tools available to you.

Do not invent information.

Do not assume that a tool was successful if
it returned an error.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.

Do not output Markdown code blocks.


FINAL RESPONSE FORMAT:

PLACEMENT-READY REPORT

Target Role:

Job Opportunities:

Current Skills:

Skill Gaps:

Recommended Projects:

GitHub Evaluation:

Placement Readiness:

Priority Action Plan:


The final response must be understandable
to a college student.

Use paragraphs and bullet points where useful.

The final response must contain actual information
obtained from the student's resume, job search,
skill-gap analysis, project recommendations and
GitHub evaluation.
"""


# ============================================================
# 9. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 10. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    resume_text: str = Field(
        description=(
            "Text extracted from the student's "
            "Resume PDF"
        )
    )

    target_role: str = Field(
        description=(
            "Target placement role, for example "
            "Full Stack Developer"
        )
    )

    github_id: str = Field(
        description=(
            "Student's public GitHub username"
        )
    )


# ============================================================
# 11. FORMAT INPUT FOR GEMMA AGENT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):

        resume_text = x.get(
            "resume_text",
            ""
        )

        target_role = x.get(
            "target_role",
            ""
        )

        github_id = x.get(
            "github_id",
            ""
        )

    else:

        resume_text = getattr(
            x,
            "resume_text",
            ""
        )

        target_role = getattr(
            x,
            "target_role",
            ""
        )

        github_id = getattr(
            x,
            "github_id",
            ""
        )


    user_input = f"""

STUDENT PLACEMENT INPUT

==================================================
RESUME
==================================================

{resume_text}


==================================================
TARGET ROLE
==================================================

{target_role}


==================================================
GITHUB ID
==================================================

{github_id}


==================================================
TASK
==================================================

Analyze this student for campus placements.

You are the Agent Core.

Select and use the appropriate tools.

Follow the complete workflow:

Job Opportunities
        ↓
Skill Gap Analysis
        ↓
Project Recommendation
        ↓
GitHub Evaluation
        ↓
Final Placement Synthesis

Return the final placement report as readable text.
"""


    return {

        "messages": [

            (
                "user",
                user_input
            )

        ]

    }


# ============================================================
# 12. EXTRACT FINAL TEXT
# ============================================================

def extract_text_response(
    agent_output
) -> str:

    if isinstance(
        agent_output,
        str
    ):

        return agent_output.strip()


    if not isinstance(
        agent_output,
        dict
    ):

        return str(agent_output)


    messages = agent_output.get(
        "messages"
    )


    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(
                    value,
                    dict
                )
                and
                "messages" in value
            ):

                messages = value[
                    "messages"
                ]

                break


    if not messages:

        return (
            "No final placement report "
            "was generated."
        )


    for message in reversed(
        messages
    ):

        # Ignore tool messages
        if message.__class__.__name__ != "AIMessage":

            continue


        content = getattr(
            message,
            "content",
            ""
        )


        if isinstance(
            content,
            str
        ):

            text = content.strip()

            if text:

                return text


        if isinstance(
            content,
            list
        ):

            text_parts = []


            for item in content:

                if isinstance(
                    item,
                    dict
                ):

                    item_type = item.get(
                        "type",
                        ""
                    )


                    # Remove Gemini thinking/reasoning
                    if item_type in {
                        "thinking",
                        "reasoning"
                    }:

                        continue


                    if item_type in {
                        "text",
                        "output_text"
                    }:

                        text = item.get(
                            "text",
                            ""
                        )

                        if text:

                            text_parts.append(
                                text
                            )

                elif isinstance(
                    item,
                    str
                ):

                    text_parts.append(
                        item
                    )


            final_text = "\n".join(
                text_parts
            ).strip()


            if final_text:

                return final_text


    return (
        "No final readable placement "
        "report was generated."
    )


# ============================================================
# 13. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "Placement-Ready AI Agent API"
    ),

    version="1.0",

    description=(
        "Placement-Ready AI Agent using "
        "LangChain, Gemma-4-31B-IT, "
        "DuckDuckGo, GitHub API, FastAPI "
        "and LangServe."
    )

)


# ============================================================
# 15. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/job-agent"

)


# ============================================================
# 16. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Placement-Ready AI Agent is running.",

        "agent_core":
            "gemma-4-31b-it",

        "endpoint":
            "/job-agent",

        "playground":
            "/job-agent/playground/"

    }


# ============================================================
# 17. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            8000
        )

    )

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=port

        )
