from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional
import anthropic
import os
import json
import uuid
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
anthropic_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global anthropic_client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    logger.info("Anthropic client initialized successfully")
    yield
    # Shutdown
    logger.info("Application shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Plan Generator API",
    description="AI-powered marketing plan generation using Claude",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://localhost:4173",  # Vite preview
        "https://*.netlify.app",  # Netlify deployments
        "https://*.vercel.app",   # Vercel deployments
        "https://*.surge.sh",     # Surge deployments
        "*"  # Allow all origins (be more restrictive in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic Models
class QuestionnaireResponse(BaseModel):
    questionId: str
    answer: Union[str, List[str]]

class GeneratePlanRequest(BaseModel):
    responses: List[QuestionnaireResponse]

class BusinessContext(BaseModel):
    industry: str
    businessModel: str
    companySize: str
    challenges: List[str]

class MarketingSquare(BaseModel):
    title: str
    summary: str
    keyPoints: List[str]
    recommendations: List[str]

class MarketingPlan(BaseModel):
    businessContext: BusinessContext
    squares: Dict[int, MarketingSquare]
    generatedAt: str
    planId: str

class SavePlanRequest(BaseModel):
    plan: MarketingPlan
    planName: Optional[str] = None

class PlanSummary(BaseModel):
    planId: str
    planName: str
    createdAt: str
    businessContext: BusinessContext

# In-memory storage (replace with database in production)
plans_storage: Dict[str, MarketingPlan] = {}

# Marketing Squares Configuration
MARKETING_SQUARES = {
    1: {
        "title": "Target Market & Customer Avatar",
        "description": "Define your ideal customers, their demographics, pain points, and buying behavior"
    },
    2: {
        "title": "Value Proposition & Messaging", 
        "description": "Craft compelling value propositions and key messages that resonate"
    },
    3: {
        "title": "Media Channels & Reach",
        "description": "Select the right marketing channels and develop content strategies"
    },
    4: {
        "title": "Lead Capture & Acquisition",
        "description": "Create effective lead magnets and conversion tactics"
    },
    5: {
        "title": "Lead Nurturing & Relationship Building",
        "description": "Develop systems to nurture leads and build lasting relationships"
    },
    6: {
        "title": "Sales Conversion & Closing",
        "description": "Optimize your sales process and closing techniques"
    },
    7: {
        "title": "Customer Experience & Delivery",
        "description": "Ensure exceptional service delivery and customer satisfaction"
    },
    8: {
        "title": "Lifetime Value & Growth",
        "description": "Maximize customer retention and increase lifetime value"
    },
    9: {
        "title": "Referral System & Advocacy",
        "description": "Build systems for referrals and turn customers into advocates"
    }
}

def get_anthropic_client():
    """Dependency to get Anthropic client"""
    if anthropic_client is None:
        raise HTTPException(status_code=500, detail="Anthropic client not initialized")
    return anthropic_client

def extract_business_context(responses: List[QuestionnaireResponse]) -> BusinessContext:
    """Extract business context from questionnaire responses"""
    response_dict = {r.questionId: r.answer for r in responses}
    
    return BusinessContext(
        industry=response_dict.get('business-industry', 'Unknown'),
        businessModel=response_dict.get('business-model', 'Unknown'),
        companySize=response_dict.get('company-size', 'Unknown'),
        challenges=response_dict.get('primary-challenges', []) if isinstance(response_dict.get('primary-challenges'), list) else []
    )

def create_claude_prompt(responses: List[QuestionnaireResponse], square_id: int) -> str:
    """Create a detailed prompt for Claude to generate marketing square content"""
    
    response_dict = {r.questionId: r.answer for r in responses}
    square_info = MARKETING_SQUARES[square_id]
    
    # Build context from responses
    business_context = f"""
Business Industry: {response_dict.get('business-industry', 'Not specified')}
Business Model: {response_dict.get('business-model', 'Not specified')}
Company Size: {response_dict.get('company-size', 'Not specified')}
Geographic Scope: {response_dict.get('geographic-scope', 'Not specified')}
Years in Operation: {response_dict.get('years-operation', 'Not specified')}
Marketing Budget: {response_dict.get('marketing-budget', 'Not specified')}
Primary Challenges: {response_dict.get('primary-challenges', 'Not specified')}
"""

    # Square-specific context
    square_context = ""
    if square_id == 1:  # Target Market
        square_context = f"""
Target Age: {response_dict.get('target-age', 'Not specified')}
Target Income: {response_dict.get('target-income', 'Not specified')}
Target Location: {response_dict.get('target-location', 'Not specified')}
Customer Pain Points: {response_dict.get('customer-pain-points', 'Not specified')}
Customer Goals: {response_dict.get('customer-goals', 'Not specified')}
Buying Behavior: {response_dict.get('buying-behavior', 'Not specified')}
"""
    elif square_id == 2:  # Value Proposition
        square_context = f"""
Unique Selling Points: {response_dict.get('unique-selling-points', 'Not specified')}
Key Benefits: {response_dict.get('key-benefits', 'Not specified')}
Brand Personality: {response_dict.get('brand-personality', 'Not specified')}
Elevator Pitch: {response_dict.get('elevator-pitch', 'Not specified')}
Messaging Tone: {response_dict.get('messaging-tone', 'Not specified')}
"""
    elif square_id == 3:  # Media Channels
        square_context = f"""
Preferred Channels: {response_dict.get('preferred-channels', 'Not specified')}
Content Types: {response_dict.get('content-types', 'Not specified')}
Content Frequency: {response_dict.get('content-frequency', 'Not specified')}
Social Platforms: {response_dict.get('social-platforms', 'Not specified')}
Reach Goals: {response_dict.get('reach-goals', 'Not specified')}
"""
    elif square_id == 4:  # Lead Capture
        square_context = f"""
Lead Magnets: {response_dict.get('lead-magnets', 'Not specified')}
Landing Pages: {response_dict.get('landing-pages', 'Not specified')}
Conversion Tactics: {response_dict.get('conversion-tactics', 'Not specified')}
Lead Goals: {response_dict.get('lead-goals', 'Not specified')}
"""
    elif square_id == 5:  # Lead Nurturing
        square_context = f"""
Email Sequences: {response_dict.get('email-sequences', 'Not specified')}
Nurturing Content: {response_dict.get('nurturing-content', 'Not specified')}
Relationship Building: {response_dict.get('relationship-building', 'Not specified')}
Nurturing Timeline: {response_dict.get('nurturing-timeline', 'Not specified')}
"""
    elif square_id == 6:  # Sales Conversion
        square_context = f"""
Sales Process: {response_dict.get('sales-process', 'Not specified')}
Common Objections: {response_dict.get('common-objections', 'Not specified')}
Closing Techniques: {response_dict.get('closing-techniques', 'Not specified')}
Conversion Rate: {response_dict.get('conversion-rate', 'Not specified')}
"""
    elif square_id == 7:  # Customer Experience
        square_context = f"""
Service Delivery: {response_dict.get('service-delivery', 'Not specified')}
Onboarding Process: {response_dict.get('onboarding-process', 'Not specified')}
Customer Support: {response_dict.get('customer-support', 'Not specified')}
Satisfaction Measurement: {response_dict.get('satisfaction-measurement', 'Not specified')}
"""
    elif square_id == 8:  # Lifetime Value
        square_context = f"""
Repeat Business: {response_dict.get('repeat-business', 'Not specified')}
Upsell Opportunities: {response_dict.get('upsell-opportunities', 'Not specified')}
Retention Strategies: {response_dict.get('retention-strategies', 'Not specified')}
Lifetime Value: {response_dict.get('lifetime-value', 'Not specified')}
"""
    elif square_id == 9:  # Referral System
        square_context = f"""
Referral Percentage: {response_dict.get('referral-percentage', 'Not specified')}
Referral Process: {response_dict.get('referral-process', 'Not specified')}
Referral Incentives: {response_dict.get('referral-incentives', 'Not specified')}
Advocacy Opportunities: {response_dict.get('advocacy-opportunities', 'Not specified')}
Word of Mouth: {response_dict.get('word-of-mouth', 'Not specified')}
"""

    prompt = f"""
You are a senior marketing strategist creating a comprehensive marketing plan. Based on the business information and questionnaire responses provided, generate detailed content for the "{square_info['title']}" section of a 9-square marketing framework.

BUSINESS CONTEXT:
{business_context}

SQUARE FOCUS: {square_info['title']}
DESCRIPTION: {square_info['description']}

RELEVANT RESPONSES:
{square_context}

Please provide a JSON response with the following structure:
{{
    "title": "{square_info['title']}",
    "summary": "A concise 1-2 sentence summary of this marketing square for this specific business",
    "keyPoints": [
        "4-6 specific, actionable key points based on the questionnaire responses",
        "Each point should be tailored to this business's specific situation",
        "Include specific data/responses where relevant",
        "Focus on insights derived from the provided information"
    ],
    "recommendations": [
        "4-6 specific, actionable recommendations for this business",
        "Each recommendation should be practical and implementable",
        "Prioritize recommendations based on the business context",
        "Include specific tactics, tools, or strategies where appropriate"
    ]
}}

Make sure all content is:
1. Specific to this business based on their responses
2. Actionable and practical
3. Professional but accessible
4. Focused on the specific marketing square's objectives
5. Consistent with their industry, size, and business model

Return only valid JSON without any additional text or formatting.
"""
    
    return prompt

async def generate_square_content(
    responses: List[QuestionnaireResponse], 
    square_id: int,
    client: anthropic.Anthropic
) -> MarketingSquare:
    """Generate content for a specific marketing square using Claude"""
    
    try:
        prompt = create_claude_prompt(responses, square_id)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse Claude's response
        response_text = message.content[0].text
        logger.info(f"Claude response for square {square_id}: {response_text[:200]}...")
        
        # Parse JSON response
        square_data = json.loads(response_text)
        
        return MarketingSquare(
            title=square_data["title"],
            summary=square_data["summary"],
            keyPoints=square_data["keyPoints"],
            recommendations=square_data["recommendations"]
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response for square {square_id}: {e}")
        # Fallback content
        return MarketingSquare(
            title=MARKETING_SQUARES[square_id]["title"],
            summary=f"AI-generated content for {MARKETING_SQUARES[square_id]['title']} based on your business profile.",
            keyPoints=[
                "Analysis of your specific business context",
                "Tailored insights based on your responses",
                "Strategic recommendations for this area",
                "Implementation considerations"
            ],
            recommendations=[
                "Develop a structured approach to this marketing area",
                "Implement tracking and measurement systems",
                "Regular review and optimization",
                "Align with overall business objectives"
            ]
        )
    except Exception as e:
        logger.error(f"Error generating content for square {square_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content for square {square_id}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Marketing Plan Generator API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "anthropic_available": anthropic_client is not None
    }

@app.post("/api/generate-plan", response_model=Dict[str, Any])
async def generate_plan(
    request: GeneratePlanRequest,
    client: anthropic.Anthropic = Depends(get_anthropic_client)
):
    """Generate a complete marketing plan using Claude AI"""
    
    try:
        logger.info(f"Generating plan for {len(request.responses)} responses")
        
        # Extract business context
        business_context = extract_business_context(request.responses)
        
        # Generate content for all 9 squares
        squares = {}
        for square_id in range(1, 10):
            logger.info(f"Generating content for square {square_id}")
            square_content = await generate_square_content(request.responses, square_id, client)
            squares[square_id] = square_content
        
        # Create the marketing plan
        plan_id = str(uuid.uuid4())
        plan = MarketingPlan(
            businessContext=business_context,
            squares=squares,
            generatedAt=datetime.utcnow().isoformat(),
            planId=plan_id
        )
        
        # Store the plan
        plans_storage[plan_id] = plan
        
        logger.info(f"Successfully generated plan {plan_id}")
        
        return {
            "plan": plan.dict(),
            "planId": plan_id,
            "message": "Marketing plan generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate marketing plan: {str(e)}")

@app.get("/api/plans", response_model=List[PlanSummary])
async def get_plans():
    """Get all saved marketing plans"""
    try:
        summaries = []
        for plan_id, plan in plans_storage.items():
            summary = PlanSummary(
                planId=plan_id,
                planName=f"Marketing Plan - {plan.businessContext.industry}",
                createdAt=plan.generatedAt,
                businessContext=plan.businessContext
            )
            summaries.append(summary)
        
        return summaries
    except Exception as e:
        logger.error(f"Error fetching plans: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch plans")

@app.get("/api/plans/{plan_id}", response_model=MarketingPlan)
async def get_plan(plan_id: str):
    """Get a specific marketing plan"""
    if plan_id not in plans_storage:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    return plans_storage[plan_id]

@app.post("/api/plans", response_model=Dict[str, str])
async def save_plan(request: SavePlanRequest):
    """Save a marketing plan"""
    try:
        plan_id = request.plan.planId or str(uuid.uuid4())
        plans_storage[plan_id] = request.plan
        
        return {
            "planId": plan_id,
            "message": "Plan saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving plan: {e}")
        raise HTTPException(status_code=500, detail="Failed to save plan")

@app.delete("/api/plans/{plan_id}", response_model=Dict[str, str])
async def delete_plan(plan_id: str):
    """Delete a marketing plan"""
    if plan_id not in plans_storage:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    del plans_storage[plan_id]
    return {"message": "Plan deleted successfully"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Marketing Plan Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)