from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import anthropic
from datetime import datetime
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MarketingPlan AI API", version="1.0.0")

# Enhanced CORS middleware for better frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models
class QuestionnaireResponse(BaseModel):
    question_id: str
    answer: Union[str, List[str]]

class UserData(BaseModel):
    id: str
    responses: List[QuestionnaireResponse]
    current_square: int
    current_question_index: int
    completed_squares: List[int]
    created_at: str
    updated_at: str

class MarketingPlanRequest(BaseModel):
    user_id: str
    responses: List[QuestionnaireResponse]

class MarketingSquare(BaseModel):
    title: str
    summary: str
    key_points: List[str]
    recommendations: List[str]

class BusinessContext(BaseModel):
    industry: str
    business_model: str
    company_size: str
    challenges: List[str]
    years_in_operation: Optional[str] = "Not specified"
    geographic_scope: Optional[str] = "Not specified"
    marketing_maturity: Optional[str] = "Not specified"
    marketing_budget: Optional[str] = "Not specified"
    time_availability: Optional[str] = "Not specified"
    business_goals: Optional[List[str]] = []

class MarketingPlan(BaseModel):
    business_context: BusinessContext
    squares: Dict[int, MarketingSquare]
    generated_at: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    api_configured: bool
    environment: str

# Initialize Anthropic client
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment")
        return None
    return anthropic.Anthropic(api_key=api_key)

# In-memory storage (replace with database in production)
user_data_store = {}
plans_store = {}

@app.get("/")
async def root():
    return {
        "message": "MarketingPlan AI API is running", 
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.post("/api/save-response")
async def save_response(response: QuestionnaireResponse):
    """Save individual questionnaire response"""
    try:
        # In production, save to database
        logger.info(f"Saving response for question: {response.question_id}")
        return {"message": "Response saved successfully", "question_id": response.question_id}
    except Exception as e:
        logger.error(f"Error saving response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save response")

@app.post("/api/save-user-data")
async def save_user_data(user_data: UserData):
    """Save user progress"""
    try:
        user_data_store[user_data.id] = user_data.dict()
        logger.info(f"Saved user data for user: {user_data.id}")
        return {"message": "User data saved successfully", "user_id": user_data.id}
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save user data")

@app.get("/api/user-data/{user_id}")
async def get_user_data(user_id: str):
    """Get user progress"""
    try:
        if user_id not in user_data_store:
            raise HTTPException(status_code=404, detail="User data not found")
        return {"user_data": user_data_store[user_id]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user data")

@app.post("/api/generate-plan")
async def generate_marketing_plan(request: MarketingPlanRequest):
    """Generate AI-powered marketing plan using Claude"""
    try:
        logger.info(f"Generating marketing plan for user: {request.user_id}")
        
        # Extract business context from responses
        business_context = extract_business_context(request.responses)
        logger.info(f"Extracted business context: {business_context}")
        
        # Generate plan using Claude AI
        ai_plan = await generate_claude_plan(request.responses, business_context)
        
        # Save plan to storage
        plans_store[request.user_id] = ai_plan.dict()
        logger.info(f"Generated and saved marketing plan for user: {request.user_id}")
        
        return ai_plan
    
    except Exception as e:
        logger.error(f"Failed to generate plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {str(e)}")

@app.get("/api/plan/{user_id}")
async def get_marketing_plan(user_id: str):
    """Retrieve saved marketing plan"""
    try:
        if user_id not in plans_store:
            raise HTTPException(status_code=404, detail="Marketing plan not found")
        return {"plan": plans_store[user_id]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving marketing plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve marketing plan")

async def generate_claude_plan(responses: List[QuestionnaireResponse], business_context: Dict) -> MarketingPlan:
    """Use Claude to generate intelligent marketing plan"""
    
    client = get_anthropic_client()
    if not client:
        logger.warning("Claude client not available, using fallback")
        return generate_fallback_plan(responses, business_context)
    
    # Prepare the prompt with user responses
    prompt = create_marketing_plan_prompt(responses, business_context)
    
    try:
        logger.info("Calling Claude API for plan generation")
        
        # Using Claude-3-Sonnet
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.7,
            system="You are a marketing strategy expert. Generate comprehensive, actionable marketing plans based on the 9-square marketing framework. Always respond with valid JSON that matches the required structure exactly.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        ai_content = message.content[0].text
        logger.info("Received response from Claude API")
        
        structured_plan = parse_ai_response_to_plan(ai_content, business_context)
        return structured_plan
    
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        # Fallback to template-based generation if AI fails
        return generate_fallback_plan(responses, business_context)

def create_marketing_plan_prompt(responses: List[QuestionnaireResponse], business_context: Dict) -> str:
    """Create a detailed prompt for Claude plan generation"""
    
    # Organize responses by square for better context
    responses_by_square = {}
    business_context_responses = []
    
    for resp in responses:
        # Check if it's a business context question (square 0) or specific square question
        if resp.question_id in ["industry", "business-model", "company-size", "years-in-operation", 
                               "geographic-scope", "primary-challenges", "marketing-maturity", 
                               "marketing-budget", "time-availability", "business-goals"]:
            business_context_responses.append(f"• {resp.question_id}: {resp.answer}")
        else:
            # Try to determine square from question content or ID patterns
            square_num = determine_square_from_question(resp.question_id)
            if square_num not in responses_by_square:
                responses_by_square[square_num] = []
            responses_by_square[square_num].append(f"• {resp.question_id}: {resp.answer}")
    
    business_context_text = "\n".join(business_context_responses)
    
    square_responses_text = ""
    for square_num in sorted(responses_by_square.keys()):
        square_responses_text += f"\nSquare {square_num} Responses:\n" + "\n".join(responses_by_square[square_num])
    
    prompt = f"""
    Based on the comprehensive business information and questionnaire responses below, create a highly personalized marketing plan using the 9-square marketing framework:

    BUSINESS CONTEXT:
    - Industry: {business_context.get('industry', 'Not specified')}
    - Business Model: {business_context.get('business_model', 'Not specified')}
    - Company Size: {business_context.get('company_size', 'Not specified')}
    - Years in Operation: {business_context.get('years_in_operation', 'Not specified')}
    - Geographic Scope: {business_context.get('geographic_scope', 'Not specified')}
    - Marketing Maturity: {business_context.get('marketing_maturity', 'Not specified')}
    - Marketing Budget: {business_context.get('marketing_budget', 'Not specified')}
    - Time Availability: {business_context.get('time_availability', 'Not specified')}
    - Key Challenges: {', '.join(business_context.get('challenges', ['Not specified']))}
    - Business Goals: {', '.join(business_context.get('business_goals', ['Not specified']))}

    BUSINESS CONTEXT RESPONSES:
    {business_context_text}
    
    DETAILED QUESTIONNAIRE RESPONSES:
    {square_responses_text}
    
    Generate a comprehensive marketing plan with these 9 squares, making each highly specific to their business context and responses:

    1. Target Market & Customer Avatar - Use their demographic, pain point, and customer behavior responses
    2. Value Proposition & Messaging - Incorporate their unique advantages and proof points
    3. Media Channels & Reach - Consider their current channels, preferences, and content capacity
    4. Lead Capture & Acquisition - Build on their current lead generation and website optimization
    5. Lead Nurturing & Relationship Building - Factor in their follow-up process and timeline
    6. Sales Conversion & Closing - Address their sales process and common objections
    7. Customer Experience & Delivery - Enhance their delivery and support methods
    8. Lifetime Value & Growth - Leverage their retention and upsell opportunities
    9. Referral System & Advocacy - Build on their current referral patterns and incentives
    
    For each square, provide:
    - A clear, specific title
    - A practical summary (2-3 sentences) tailored to their exact business situation
    - 4-5 key insights based on their specific responses, industry, and context
    - 4-5 highly actionable recommendations they can implement given their budget, time, and maturity level
    
    Make every recommendation specific to:
    - Their industry and business model
    - Their budget constraints ({business_context.get('marketing_budget', 'Not specified')})
    - Their time availability ({business_context.get('time_availability', 'Not specified')})
    - Their marketing maturity level ({business_context.get('marketing_maturity', 'Not specified')})
    - Their specific challenges and goals
    
    CRITICAL: Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
    {{
        "squares": {{
            "1": {{
                "title": "Target Market & Customer Avatar",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "2": {{
                "title": "Value Proposition & Messaging",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "3": {{
                "title": "Media Channels & Reach",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "4": {{
                "title": "Lead Capture & Acquisition",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "5": {{
                "title": "Lead Nurturing & Relationship Building",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "6": {{
                "title": "Sales Conversion & Closing",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "7": {{
                "title": "Customer Experience & Delivery",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "8": {{
                "title": "Lifetime Value & Growth",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }},
            "9": {{
                "title": "Referral System & Advocacy",
                "summary": "Brief summary here tailored to their responses",
                "key_points": ["Point 1 based on their data", "Point 2", "Point 3", "Point 4"],
                "recommendations": ["Specific rec 1", "Specific rec 2", "Specific rec 3", "Specific rec 4"]
            }}
        }}
    }}
    """
    
    return prompt

def determine_square_from_question(question_id: str) -> int:
    """Determine which marketing square a question belongs to based on question ID patterns"""
    question_id_lower = question_id.lower()
    
    # Square 1: Target Market & Customer Avatar
    if any(keyword in question_id_lower for keyword in ['target', 'demographics', 'customer', 'pain-points', 'goals', 'buying-behavior', 'sources']):
        return 1
    
    # Square 2: Value Proposition & Messaging  
    elif any(keyword in question_id_lower for keyword in ['problem-solved', 'unique-advantages', 'benefits', 'emotional', 'proof-points']):
        return 2
    
    # Square 3: Media Channels & Reach
    elif any(keyword in question_id_lower for keyword in ['marketing-channels', 'digital-marketing', 'content-creation']):
        return 3
    
    # Square 4: Lead Capture & Acquisition
    elif any(keyword in question_id_lower for keyword in ['lead-generation', 'lead-magnets', 'website-optimization']):
        return 4
    
    # Square 5: Lead Nurturing & Relationship Building
    elif any(keyword in question_id_lower for keyword in ['follow-up', 'email-marketing', 'educational-content', 'relationship-timeline']):
        return 5
    
    # Square 6: Sales Conversion & Closing
    elif any(keyword in question_id_lower for keyword in ['sales-process', 'objections', 'pricing-strategy', 'conversion-rate']):
        return 6
    
    # Square 7: Customer Experience & Delivery
    elif any(keyword in question_id_lower for keyword in ['delivery-method', 'onboarding', 'customer-support', 'feedback']):
        return 7
    
    # Square 8: Lifetime Value & Growth
    elif any(keyword in question_id_lower for keyword in ['repeat-business', 'upsell', 'retention', 'customer-value']):
        return 8
    
    # Square 9: Referral System & Advocacy
    elif any(keyword in question_id_lower for keyword in ['referrals', 'referral-process', 'incentives', 'advocacy']):
        return 9
    
    # Default to general if can't determine
    else:
        return 0

def extract_business_context(responses: List[QuestionnaireResponse]) -> Dict:
    """Extract comprehensive business context from all responses"""
    context = {
        "industry": "Not specified",
        "business_model": "Not specified", 
        "company_size": "Not specified",
        "challenges": [],
        "years_in_operation": "Not specified",
        "geographic_scope": "Not specified",
        "marketing_maturity": "Not specified",
        "marketing_budget": "Not specified",
        "time_availability": "Not specified",
        "business_goals": []
    }
    
    # Map all the business context questions from your frontend
    for response in responses:
        question_id = response.question_id.lower()
        answer = response.answer
        
        # Business Context Questions (Square 0)
        if question_id == "industry":
            context["industry"] = answer
        elif question_id == "business-model":
            context["business_model"] = answer
        elif question_id == "company-size":
            context["company_size"] = answer
        elif question_id == "years-in-operation":
            context["years_in_operation"] = answer
        elif question_id == "geographic-scope":
            context["geographic_scope"] = answer
        elif question_id == "primary-challenges":
            context["challenges"] = answer if isinstance(answer, list) else [answer]
        elif question_id == "marketing-maturity":
            context["marketing_maturity"] = answer
        elif question_id == "marketing-budget":
            context["marketing_budget"] = answer
        elif question_id == "time-availability":
            context["time_availability"] = answer
        elif question_id == "business-goals":
            context["business_goals"] = answer if isinstance(answer, list) else [answer]
        
        # Legacy support for older question IDs
        elif "industry" in question_id:
            context["industry"] = answer
        elif "model" in question_id:
            context["business_model"] = answer
        elif "size" in question_id:
            context["company_size"] = answer
        elif "challenge" in question_id:
            challenges = answer if isinstance(answer, list) else [answer]
            context["challenges"].extend(challenges)
    
    return context

def parse_ai_response_to_plan(ai_content: str, business_context: Dict) -> MarketingPlan:
    """Parse Claude response into structured MarketingPlan"""
    try:
        # Clean the response - remove any markdown formatting
        cleaned_content = ai_content.strip()
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith('```'):
            cleaned_content = cleaned_content[:-3]
        cleaned_content = cleaned_content.strip()
        
        logger.info(f"Parsing Claude response (length: {len(cleaned_content)})")
        
        # Parse JSON
        parsed_content = json.loads(cleaned_content)
        
        squares = {}
        for square_id, square_data in parsed_content.get("squares", {}).items():
            squares[int(square_id)] = MarketingSquare(
                title=square_data.get("title", f"Square {square_id}"),
                summary=square_data.get("summary", "Summary not available"),
                key_points=square_data.get("key_points", []),
                recommendations=square_data.get("recommendations", [])
            )
        
        logger.info(f"Successfully parsed {len(squares)} marketing squares")
        
        return MarketingPlan(
            business_context=BusinessContext(
                industry=business_context.get("industry", "Not specified"),
                business_model=business_context.get("business_model", "Not specified"),
                company_size=business_context.get("company_size", "Not specified"),
                challenges=business_context.get("challenges", []),
                years_in_operation=business_context.get("years_in_operation", "Not specified"),
                geographic_scope=business_context.get("geographic_scope", "Not specified"),
                marketing_maturity=business_context.get("marketing_maturity", "Not specified"),
                marketing_budget=business_context.get("marketing_budget", "Not specified"),
                time_availability=business_context.get("time_availability", "Not specified"),
                business_goals=business_context.get("business_goals", [])
            ),
            squares=squares,
            generated_at=datetime.now().isoformat()
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Claude response: {ai_content[:500]}...")
        # Fallback to template-based generation
        return generate_fallback_plan([], business_context)
    except Exception as e:
        logger.error(f"Error parsing Claude response: {e}")
        return generate_fallback_plan([], business_context)

def generate_fallback_plan(responses: List[QuestionnaireResponse], business_context: Dict) -> MarketingPlan:
    """Generate a basic template plan if Claude fails"""
    logger.info("Generating fallback marketing plan")
    
    squares = {}
    
    square_templates = {
        1: {
            "title": "Target Market & Customer Avatar",
            "summary": f"Define your ideal customers in the {business_context.get('industry', 'your')} industry with a focus on {business_context.get('business_model', 'your business model')}.",
            "key_points": [
                f"Operating in {business_context.get('industry', 'your industry')} sector",
                f"Business model: {business_context.get('business_model', 'needs definition')}",
                f"Company size: {business_context.get('company_size', 'needs assessment')}",
                "Customer segmentation is critical for targeted marketing"
            ],
            "recommendations": [
                "Create detailed buyer personas with demographics and psychographics",
                "Conduct customer interviews to validate assumptions",
                "Analyze competitor customer bases for market insights",
                "Use Google Analytics and social media insights for data"
            ]
        },
        2: {
            "title": "Value Proposition & Messaging",
            "summary": "Develop compelling messages that clearly communicate your unique value and resonate with your target market's needs.",
            "key_points": [
                "Need clear differentiation from competitors",
                "Message must address specific customer pain points",
                "Brand personality should align with target audience",
                "Consistent messaging across all touchpoints is essential"
            ],
            "recommendations": [
                "Test different value propositions with target customers",
                "Create a comprehensive brand style and voice guide",
                "Develop elevator pitches for different scenarios",
                "A/B test messaging across marketing channels"
            ]
        },
        3: {
            "title": "Media Channels & Reach",
            "summary": "Select and optimize the right marketing channels where your target audience is most active and engaged.",
            "key_points": [
                "Multi-channel approach increases reach and frequency",
                "Content strategy must align with channel characteristics",
                "Consistent publishing schedule builds audience trust",
                "Channel selection based on audience behavior data"
            ],
            "recommendations": [
                "Start with 2-3 channels and master them first",
                "Create a content calendar with channel-specific content",
                "Repurpose content across channels with platform optimization",
                "Track performance metrics for each channel monthly"
            ]
        },
        4: {
            "title": "Lead Capture & Acquisition",
            "summary": "Implement systems to attract and capture qualified leads through strategic touchpoints and compelling offers.",
            "key_points": [
                "Lead magnets must provide genuine value to prospects",
                "Multiple capture points increase conversion opportunities",
                "Landing page optimization is crucial for conversions",
                "Lead quality is more important than quantity"
            ],
            "recommendations": [
                "Create industry-specific lead magnets (ebooks, tools, templates)",
                "Optimize landing pages with clear value propositions",
                "Implement progressive profiling to gather lead information",
                "Set up lead scoring to prioritize follow-up efforts"
            ]
        },
        5: {
            "title": "Lead Nurturing & Relationship Building",
            "summary": "Develop systematic approaches to educate and engage leads through their buyer journey until they're sales-ready.",
            "key_points": [
                "Automated email sequences provide consistent touchpoints",
                "Educational content builds trust and authority",
                "Personalization increases engagement rates significantly",
                "Multi-touch nurturing campaigns improve conversion rates"
            ],
            "recommendations": [
                "Create email drip campaigns based on lead behavior",
                "Segment leads by interest, industry, or buying stage",
                "Provide valuable educational content consistently",
                "Use marketing automation to scale personalization"
            ]
        },
        6: {
            "title": "Sales Conversion & Closing",
            "summary": "Optimize your sales process to effectively convert qualified leads into paying customers with clear systems and follow-up.",
            "key_points": [
                "Clear sales process reduces friction and confusion",
                "Follow-up systems prevent leads from falling through cracks",
                "Objection handling preparation improves close rates",
                "Sales and marketing alignment is crucial for success"
            ],
            "recommendations": [
                "Document and standardize your sales process steps",
                "Create objection handling scripts and resources",
                "Implement CRM system for lead tracking and follow-up",
                "Establish clear handoff process between marketing and sales"
            ]
        },
        7: {
            "title": "Customer Experience & Delivery",
            "summary": "Ensure exceptional customer experience and service delivery that exceeds expectations and builds loyalty.",
            "key_points": [
                "First impressions set the tone for entire relationship",
                "Consistent delivery builds trust and reduces churn",
                "Customer feedback loops improve service quality",
                "Proactive communication prevents issues and builds confidence"
            ],
            "recommendations": [
                "Create customer onboarding process with clear expectations",
                "Implement regular check-ins and progress updates",
                "Establish customer feedback collection and response system",
                "Document service delivery standards for consistency"
            ]
        },
        8: {
            "title": "Lifetime Value & Growth",
            "summary": "Maximize customer lifetime value through retention strategies, upselling, and expanding relationships over time.",
            "key_points": [
                "Retention costs significantly less than acquisition",
                "Upselling to existing customers has higher success rates",
                "Customer success directly impacts lifetime value",
                "Regular value delivery maintains customer relationships"
            ],
            "recommendations": [
                "Create customer success programs focused on outcomes",
                "Develop upsell and cross-sell opportunity mapping",
                "Implement customer health scoring and intervention triggers",
                "Establish regular business reviews with key accounts"
            ]
        },
        9: {
            "title": "Referral System & Advocacy",
            "summary": "Build systematic approaches to generate referrals and turn satisfied customers into active brand advocates.",
            "key_points": [
                "Satisfied customers are often willing to refer but need prompting",
                "Referral systems require clear processes and incentives",
                "Customer testimonials and case studies build social proof",
                "Word-of-mouth marketing has highest trust and conversion rates"
            ],
            "recommendations": [
                "Create formal referral program with clear incentives",
                "Systematically collect customer testimonials and reviews",
                "Develop case studies showcasing customer success stories",
                "Make it easy for customers to share and recommend your business"
            ]
        }
    }
    
    # Use templates for all squares
    for i in range(1, 10):
        if i in square_templates:
            squares[i] = MarketingSquare(**square_templates[i])
    
    return MarketingPlan(
        business_context=BusinessContext(
            industry=business_context.get("industry", "Not specified"),
            business_model=business_context.get("business_model", "Not specified"),
            company_size=business_context.get("company_size", "Not specified"),
            challenges=business_context.get("challenges", []),
            years_in_operation=business_context.get("years_in_operation", "Not specified"),
            geographic_scope=business_context.get("geographic_scope", "Not specified"),
            marketing_maturity=business_context.get("marketing_maturity", "Not specified"),
            marketing_budget=business_context.get("marketing_budget", "Not specified"),
            time_availability=business_context.get("time_availability", "Not specified"),
            business_goals=business_context.get("business_goals", [])
        ),
        squares=squares,
        generated_at=datetime.now().isoformat()
    )

@app.get("/api/questions")
async def get_questions_structure():
    """Get the complete questions structure for frontend reference"""
    return {
        "marketing_squares": [
            {"id": 1, "title": "Target Market & Customer Avatar", "description": "Define your ideal customer profile"},
            {"id": 2, "title": "Value Proposition & Messaging", "description": "Craft your unique value proposition"},
            {"id": 3, "title": "Media Channels & Reach", "description": "Select optimal marketing channels"},
            {"id": 4, "title": "Lead Capture & Acquisition", "description": "Design lead generation systems"},
            {"id": 5, "title": "Lead Nurturing & Relationship Building", "description": "Build relationships with prospects"},
            {"id": 6, "title": "Sales Conversion & Closing", "description": "Optimize your sales process"},
            {"id": 7, "title": "Customer Experience & Delivery", "description": "Deliver exceptional customer experience"},
            {"id": 8, "title": "Lifetime Value & Growth", "description": "Maximize customer lifetime value"},
            {"id": 9, "title": "Referral System & Advocacy", "description": "Create systematic referral generation"}
        ],
        "expected_question_ids": {
            "business_context": [
                "industry", "business-model", "company-size", "years-in-operation",
                "geographic-scope", "primary-challenges", "marketing-maturity", 
                "marketing-budget", "time-availability", "business-goals"
            ],
            "square_1": [
                "target-demographics-age", "target-demographics-income", "target-location",
                "customer-pain-points", "customer-goals", "customer-buying-behavior", 
                "current-customer-sources"
            ],
            "square_2": [
                "core-problem-solved", "unique-advantages", "tangible-benefits", 
                "emotional-benefits", "proof-points"
            ],
            "square_3": [
                "current-marketing-channels", "digital-marketing-preferences", 
                "content-creation-capacity"
            ],
            "square_4": [
                "current-lead-generation", "lead-magnets", "website-optimization"
            ],
            "square_5": [
                "follow-up-process", "email-marketing-current", "educational-content", 
                "relationship-timeline"
            ],
            "square_6": [
                "sales-process", "common-objections", "pricing-strategy", "conversion-rate"
            ],
            "square_7": [
                "delivery-method", "onboarding-process", "customer-support", "customer-feedback"
            ],
            "square_8": [
                "repeat-business", "upsell-opportunities", "customer-retention", "average-customer-value"
            ],
            "square_9": [
                "current-referrals", "referral-process", "referral-incentives", "customer-advocacy"
            ]
        }
    }

@app.post("/api/validate-responses")
async def validate_responses(responses: List[QuestionnaireResponse]):
    """Validate that responses contain expected question IDs"""
    expected_questions = await get_questions_structure()
    all_expected_ids = []
    
    for square_questions in expected_questions["expected_question_ids"].values():
        all_expected_ids.extend(square_questions)
    
    provided_ids = [resp.question_id for resp in responses]
    
    missing_ids = [qid for qid in all_expected_ids if qid not in provided_ids]
    unexpected_ids = [qid for qid in provided_ids if qid not in all_expected_ids]
    
    return {
        "is_valid": len(missing_ids) == 0,
        "total_responses": len(responses),
        "expected_questions": len(all_expected_ids),
        "missing_question_ids": missing_ids,
        "unexpected_question_ids": unexpected_ids,
        "completeness_percentage": ((len(all_expected_ids) - len(missing_ids)) / len(all_expected_ids)) * 100
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Enhanced health check for deployment monitoring"""
    api_configured = bool(os.getenv("ANTHROPIC_API_KEY"))
    environment = os.getenv("ENVIRONMENT", "development")
    
    return HealthCheck(
        status="healthy", 
        timestamp=datetime.now().isoformat(),
        api_configured=api_configured,
        environment=environment
    )

# Additional endpoints for deployment monitoring
@app.get("/api/status")
async def api_status():
    """API status endpoint for monitoring"""
    return {
        "api_status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/api/save-response",
            "/api/save-user-data", 
            "/api/user-data/{user_id}",
            "/api/generate-plan",
            "/api/plan/{user_id}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)