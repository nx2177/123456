def generate_sample_resume():
    """
    Generate a realistic sample resume for testing
    
    Returns:
        str: A sample resume in plain text format
    """
    return """
JOHN ANDERSON
Software Engineer
john.anderson@email.com | (555) 123-4567 | San Francisco, CA | linkedin.com/in/johnanderson

SUMMARY
Experienced full-stack developer with 6+ years building scalable web applications and distributed systems. 
Passionate about clean code, performance optimization, and applying machine learning to solve complex problems.

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, Java, SQL, HTML/CSS
Frameworks: React, Node.js, Django, Flask, Express.js
Cloud: AWS (EC2, S3, Lambda, DynamoDB), Google Cloud Platform
Tools: Docker, Kubernetes, Git, Jenkins, CI/CD pipelines
Data: PostgreSQL, MongoDB, Redis, Elasticsearch
Machine Learning: PyTorch, TensorFlow, scikit-learn, NLP

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | Aug 2020 - Present
- Led development of a real-time data processing pipeline handling 50M+ daily events using Kafka and Spark
- Redesigned authentication system reducing login latency by 40% and improving security compliance
- Mentored junior engineers and established code review processes improving code quality by 35%
- Implemented automated testing strategy increasing test coverage from 65% to 92%
- Collaborated with product and design teams to deliver features that increased user engagement by 28%

Software Engineer | DataSystems Solutions | Mar 2018 - Jul 2020
- Built RESTful APIs with Django and Flask that power mobile and web applications used by 200K+ users
- Optimized database queries reducing page load time by 60% and server costs by 25%
- Developed microservices architecture for improved scalability and maintainability
- Created analytics dashboard with React.js visualizing key metrics for executive decision making
- Participated in 24-hour on-call rotation maintaining 99.9% system uptime

Junior Developer | WebFront Technologies | Jun 2016 - Feb 2018
- Developed responsive web interfaces using React and Redux that improved mobile conversion by 15%
- Built automated testing suite reducing QA time by 30% and catching critical bugs before deployment
- Contributed to open-source projects including documentation improvements and bug fixes
- Modernized legacy codebase to use contemporary JavaScript standards and patterns

EDUCATION
Master of Science in Computer Science | Stanford University | 2016
Bachelor of Science in Computer Engineering | University of California, Berkeley | 2014

PROJECTS
MindfulML - Open source library for ethical AI implementation with 500+ GitHub stars
DataVisualize - Interactive data visualization tool built with D3.js and React
CloudScale - AWS infrastructure automation toolkit used by 20+ startups

SOFT SKILLS
- Strong communication skills with technical and non-technical stakeholders
- Effective team leadership and collaboration across departments
- Creative problem-solving approach with analytical thinking
- Adaptable to changing requirements and technologies
- Excellent time management handling multiple priorities
"""

def generate_sample_job_description():
    """
    Generate a realistic sample job description for testing
    
    Returns:
        str: A sample job description in plain text format
    """
    return """
SENIOR FULL-STACK ENGINEER
AI Products Team | InnovateAI Inc.

ABOUT THE ROLE
We are looking for an experienced Full-Stack Engineer to join our AI Products team. In this role, you'll work on building and scaling our machine learning-powered applications that help businesses automate document processing and extract insights from unstructured data. You'll collaborate with a cross-functional team of engineers, data scientists, and product managers to deliver high-quality software solutions.

RESPONSIBILITIES
- Design, build and maintain scalable web applications using Python, React, and cloud technologies
- Develop efficient and reliable APIs to support our machine learning models and front-end applications
- Implement and optimize database designs for performance, reliability, and scalability
- Work closely with the ML team to integrate and deploy machine learning models
- Lead technical initiatives and mentor junior engineers
- Participate in code reviews and ensure high quality standards in the codebase
- Troubleshoot and resolve complex technical issues in production environments
- Stay up-to-date with emerging technologies and industry best practices

REQUIRED SKILLS & EXPERIENCE
- 5+ years of experience in software development with focus on web applications
- Strong proficiency in Python and JavaScript/TypeScript
- Experience with modern web frameworks such as React, Angular, or Vue.js
- Expertise in building RESTful APIs and working with databases (SQL and NoSQL)
- Experience with cloud platforms (AWS, Google Cloud, or Azure)
- Solid understanding of CI/CD pipelines and DevOps practices
- Familiarity with containerization technologies like Docker and Kubernetes
- Basic understanding of machine learning concepts and NLP
- Experience with version control systems (Git) and collaborative development
- Must have US work authorization (this position does not qualify for sponsorship)

NICE TO HAVE
- Experience with real-time data processing systems (Kafka, Spark)
- Background in developing ML-powered applications
- Knowledge of GraphQL and modern API patterns
- Experience with serverless architectures
- Open-source contributions
- Security best practices implementation

EDUCATION
Bachelor's degree in Computer Science, Engineering or related field, or equivalent practical experience

SOFT SKILLS WE VALUE
- Excellent communication and collaboration abilities
- Strong problem-solving and analytical thinking
- Ability to work effectively in a fast-paced environment
- Leadership and mentoring capabilities
- Self-motivated with ability to work independently
- Adaptability and willingness to learn new technologies

BENEFITS & PERKS
- Competitive salary and equity package
- Health, dental, and vision insurance
- 401(k) matching
- Flexible working hours and remote options
- Professional development budget
- Generous paid time off
- Modern office with catered meals
- Regular team events and activities
"""

def generate_candidate_resumes():
    """
    Generate 5 candidate resumes with specific distribution:
    - 1 clear mismatch
    - 1 strong match
    - 3 moderate matches with different strengths
    
    Returns:
        list: List of 5 resume texts
    """
    # Resume 1: Clear mismatch (lacks US work authorization and missing core skills)
    resume1 = """
ALEX ZHANG
Software Developer
alex.zhang@email.com | (555) 987-6543 | Vancouver, Canada | linkedin.com/in/alexzhang

SUMMARY
Detail-oriented backend developer with 3 years of experience building data pipelines and API services.
Seeking opportunities to apply my skills in a challenging environment where I can grow professionally.

TECHNICAL SKILLS
Languages: C++, Java, Ruby, PHP
Frameworks: Ruby on Rails, Laravel
Cloud: Google Cloud Platform
Tools: Git, SVN, Jenkins
Data: MySQL, PostgreSQL
ML: Basic statistical analysis

WORK EXPERIENCE

Junior Software Developer | TechStartup Inc. | Jun 2021 - Present
- Built backend services for e-commerce applications using Ruby on Rails
- Optimized SQL queries improving database performance by 30%
- Implemented payment processing integration with Stripe and PayPal
- Collaborated with UX team to improve checkout flow conversion rates

Web Developer | CanadaTech Solutions | Aug 2019 - May 2021
- Developed and maintained PHP-based web applications for small businesses
- Created custom WordPress themes and plugins for client websites
- Implemented responsive design principles for mobile compatibility
- Managed hosting infrastructure for over 20 client websites

EDUCATION
Bachelor of Science in Computer Science | University of British Columbia | 2019

PROJECTS
ShopCart - E-commerce platform built with Ruby on Rails
WeatherNow - Weather forecast application using public APIs

SOFT SKILLS
- Collaborative team player
- Strong written communication
- Detail-oriented and organized
- Fast learner with strong research skills

ADDITIONAL INFORMATION
- Canadian citizen, would require sponsorship for US work
- Fluent in English and Mandarin Chinese
"""

    # Resume 2: Strong match (excellent fit with high alignment)
    resume2 = """
SARAH JOHNSON
Senior Software Engineer
sarah.johnson@email.com | (555) 234-5678 | San Francisco, CA | linkedin.com/in/sarahjohnson
U.S. Citizen

SUMMARY
Highly accomplished full-stack engineer with 8+ years of experience building scalable web applications and ML-powered systems.
Passionate about clean architecture, performance optimization, and delivering exceptional user experiences.

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, Go, SQL, HTML/CSS
Frameworks: React, Node.js, Django, Flask, Express, Next.js
Cloud: AWS (EC2, S3, Lambda, DynamoDB, SageMaker), Google Cloud Platform, Azure
Tools: Docker, Kubernetes, Git, Jenkins, GitHub Actions, CircleCI, Terraform
Data: PostgreSQL, MongoDB, Redis, Elasticsearch, Snowflake
Machine Learning: PyTorch, TensorFlow, scikit-learn, NLP, Computer Vision

WORK EXPERIENCE

Lead Software Engineer | AITech Solutions | Jan 2020 - Present
- Architected and built a scalable ML platform processing 100M+ documents monthly using AWS and TensorFlow
- Led migration from monolith to microservices, reducing deployment time by 80% and improving system resilience
- Mentored team of 8 engineers and established agile development processes
- Implemented real-time data processing pipeline using Kafka and Spark
- Reduced infrastructure costs by 40% through optimization and serverless architecture
- Collaborated with product managers and designers to create intuitive user interfaces with React

Senior Software Engineer | DataViz Inc. | Feb 2017 - Dec 2019
- Built RESTful and GraphQL APIs powering enterprise-grade applications used by Fortune 500 companies
- Led frontend development using React and TypeScript with 98% test coverage
- Implemented CI/CD pipelines reducing release cycles from weeks to days
- Created analytics dashboard visualizing complex datasets for executive decision making
- Optimized application performance improving page load time by 65%

Software Engineer | WebScale Technologies | Jun 2015 - Jan 2017
- Developed responsive web applications using modern JavaScript frameworks
- Implemented authentication and authorization systems with robust security practices
- Built automated testing suite reducing QA time by 40%
- Contributed to open-source projects including core framework improvements

EDUCATION
Master of Science in Computer Science | Stanford University | 2015
Bachelor of Science in Computer Engineering | MIT | 2013

PROJECTS
NLPEngine - Open source NLP toolkit with 1,200+ GitHub stars
CloudArchitect - Infrastructure as code templates for AWS and GCP
SecurityFirst - Authentication library with advanced security features

SOFT SKILLS
- Outstanding written and verbal communication
- Strong leadership and team mentoring capabilities
- Excellent problem-solving and analytical thinking
- Highly adaptable to changing requirements and technologies
- Exceptional time management and prioritization skills
"""

    # Resume 3: Moderate match with strength in technical skills
    resume3 = """
DAVID KIM
Full-Stack Developer
david.kim@email.com | (555) 345-6789 | Austin, TX | linkedin.com/in/davidkim
U.S. Citizen

SUMMARY
Versatile developer with 4 years of experience specializing in modern web technologies and cloud architecture.
Strong focus on technical excellence and staying current with emerging technologies.

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, Rust, Go, SQL, HTML/CSS
Frameworks: React, Vue.js, Node.js, Express, FastAPI, Svelte
Cloud: AWS (complete suite), Google Cloud, Azure
Tools: Docker, Kubernetes, Terraform, Ansible, Git, GitHub Actions, ArgoCD
Data: PostgreSQL, MongoDB, DynamoDB, Neo4j, Cassandra, Redis
Machine Learning: PyTorch, TensorFlow, Hugging Face Transformers

WORK EXPERIENCE

Software Engineer | CloudTech Systems | Mar 2021 - Present
- Implemented serverless architectures using AWS Lambda and API Gateway
- Built microservices with Kubernetes and Docker handling 30M+ daily requests
- Developed real-time data processing systems using Kafka and Kinesis
- Optimized CI/CD pipelines reducing deployment time by 70%
- Implemented GraphQL APIs for improved frontend data fetching

Junior Developer | WebApps Inc. | Jun 2019 - Feb 2021
- Developed feature-rich web applications using React and TypeScript
- Built RESTful APIs with Express.js and PostgreSQL
- Implemented automated testing strategies with Jest and Cypress
- Participated in code reviews and agile development processes

EDUCATION
Bachelor of Science in Computer Science | University of Texas | 2019

PROJECTS
ServerlessFramework - Open-source toolkit for serverless deployments
DatabaseBenchmark - Performance comparison tool for NoSQL databases
APISecure - Security audit tool for REST APIs

SOFT SKILLS
- Good written communication
- Independent worker
- Analytical problem solver
- Quick learner with new technologies

CERTIFICATIONS
- AWS Certified Solutions Architect
- Kubernetes Administrator (CKA)
- MongoDB Certified Developer
"""

    # Resume 4: Moderate match with strength in experience
    resume4 = """
MICHAEL WILSON
Software Engineering Manager
michael.wilson@email.com | (555) 456-7890 | Seattle, WA | linkedin.com/in/michaelwilson
U.S. Citizen

SUMMARY
Experienced engineering leader with 12+ years in software development and 5 years in management roles.
Track record of delivering complex projects and developing high-performance engineering teams.

TECHNICAL SKILLS
Languages: JavaScript, Python, Java, C#
Frameworks: React, Angular, Django, Spring Boot
Cloud: AWS, Azure
Tools: Git, Jenkins, JIRA, Confluence
Data: SQL Server, MySQL, MongoDB
Methodologies: Agile, Scrum, Kanban

WORK EXPERIENCE

Engineering Manager | Enterprise Systems Inc. | Jan 2019 - Present
- Led team of 15 engineers developing enterprise SaaS platform used by 200+ corporations
- Managed delivery of 4 major product releases annually with 99.9% uptime SLA
- Established code quality processes improving stability and reducing bugs by 45%
- Coordinated with product management to align technical roadmap with business objectives
- Implemented agile methodologies improving team velocity by 30%

Senior Software Engineer | DataSoft Corp | Mar 2015 - Dec 2018
- Led development of distributed data processing pipeline handling 2TB daily
- Architected migration from monolith to microservices architecture
- Mentored junior engineers and conducted technical interviews
- Implemented monitoring and alerting systems reducing incident response time by 60%

Software Engineer | TechSolutions LLC | Jun 2011 - Feb 2015
- Developed web applications using JavaScript frameworks and RESTful APIs
- Built reporting systems integrating data from multiple enterprise sources
- Optimized SQL queries and database schemas improving application performance
- Implemented automated testing frameworks to ensure code quality

EDUCATION
Bachelor of Science in Computer Science | University of Washington | 2011

SOFT SKILLS
- Exceptional leadership and team management
- Strong communication with technical and non-technical stakeholders
- Strategic planning and roadmap development
- Excellent conflict resolution and problem-solving
- Mentoring and career development

CERTIFICATIONS
- Certified Scrum Master
- Project Management Professional (PMP)
"""

    # Resume 5: Moderate match with strength in soft skills
    resume5 = """
RACHEL GARCIA
Product-Focused Developer
rachel.garcia@email.com | (555) 567-8901 | Chicago, IL | linkedin.com/in/rachelgarcia
U.S. Citizen

SUMMARY
Empathetic and collaborative developer with 5 years of experience bridging the gap between technical implementation and user needs.
Passionate about creating intuitive user experiences and translating business requirements into technical solutions.

TECHNICAL SKILLS
Languages: JavaScript, Python, HTML/CSS, SQL
Frameworks: React, Express.js, Bootstrap, Tailwind CSS
Cloud: AWS (basic), Netlify, Vercel
Tools: Git, Figma, Trello, Notion
Data: PostgreSQL, MongoDB
Testing: Jest, Cypress, user testing methods

WORK EXPERIENCE

Frontend Engineer | UserFirst Technologies | Apr 2020 - Present
- Collaborated closely with UX designers to implement pixel-perfect, accessible interfaces
- Facilitated 20+ user research sessions to gather feedback and inform development priorities
- Reduced customer support tickets by 35% through intuitive UI improvements
- Presented technical concepts to non-technical stakeholders in clear, understandable terms
- Mentored junior developers on user-centered development practices

Full-Stack Developer | WebSolutions Agency | Jul 2018 - Mar 2020
- Served as primary client liaison for 10+ projects, translating business needs into technical requirements
- Built custom web applications with focus on exceptional user experience
- Collaborated with cross-functional teams to ensure alignment between design and implementation
- Implemented analytics tracking to measure user engagement and conversion metrics

Customer Success Specialist | TechStart Inc. | Jan 2017 - Jun 2018
- Onboarded and supported enterprise customers using technical products
- Documented user pain points and collaborated with engineering to implement solutions
- Created training materials and conducted webinars for product users

EDUCATION
Bachelor of Arts in Communications | Northwestern University | 2016
Full-Stack Web Development Bootcamp | Tech Academy Chicago | 2017

PROJECTS
AccessibleUI - Component library focused on web accessibility
UserJourney - Open-source tool for mapping and analyzing user flows
FeedbackLoop - Customer feedback integration platform

SOFT SKILLS
- Outstanding verbal and written communication
- Excellent client and stakeholder management
- Strong empathy and active listening
- Exceptional presentation and documentation skills
- Highly collaborative team player with ability to work across departments
- Creative problem-solving with focus on user needs

ADDITIONAL
- Conference speaker at UX Dev Summit 2022
- Volunteer coding instructor for underrepresented groups in tech
"""

    return [resume1, resume2, resume3, resume4, resume5]

def generate_sample_pair():
    """
    Generate a sample pair of resume and matching job description
    
    Returns:
        tuple: (resume_text, job_description)
    """
    return generate_sample_resume(), generate_sample_job_description() 