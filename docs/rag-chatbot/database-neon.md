---
title: Setting up Neon Postgres Database
sidebar_label: Neon Setup
---

# Setting up Neon Serverless Postgres Free Tier

## Overview

Neon is a serverless PostgreSQL platform that provides a free tier suitable for development and small-scale applications. It offers automatic scaling, branching, and other advanced features while maintaining PostgreSQL compatibility.

## Prerequisites

- An internet connection
- A Neon account

## Creating a Neon Account

1. **Visit Neon**: Go to [neon.tech](https://neon.tech) to access the Neon platform.

2. **Sign Up**: Click on "Sign up" and create an account using your email address or GitHub/Google account.

3. **Verify Email**: Check your email for a verification message from Neon and click the verification link.

## Setting Up a Free Project

1. **Create a New Project**:
   - After logging in, click on "New Project"
   - Select the "Free" tier option (this provides 1 shared CPU, 1GB RAM, 3 databases, and 10GB of data)
   - Give your project a descriptive name (e.g., `rag-chatbot-db`)

2. **Choose Region**:
   - Select a region closest to your users for optimal performance
   - Click "Create Project"

3. **Wait for Provisioning**:
   - The project creation may take a minute or two
   - Wait until the project status shows as "Active" or "Ready"

## Getting Connection Details

1. **Access Project Dashboard**:
   - Once your project is ready, click on it to access the dashboard
   - Note down the connection string details:
     - Host
     - Database name
     - Username
     - Password

2. **Connection String Format**:
   ```
   postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
   ```

## Configuring Your Application

### Environment Variables

Set up the following environment variables in your application:

```bash
NEON_DB_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
```

### Database Schema

Your application will need to create the following tables for the RAG Chatbot:

#### Documents Table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Chat History Table
```sql
CREATE TABLE chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    context_metadata JSONB
);
```

## Using Neon Branches

Neon's branching feature allows you to create isolated environments:

1. **Create a Branch**:
   - In the Neon dashboard, go to the "Branches" tab
   - Click "Create Branch"
   - Give it a name (e.g., `development`, `staging`)

2. **Switch Between Branches**:
   - Each branch has its own connection string
   - Use different environment variables for different branches

## Connection Pooling

For production applications, consider using connection pooling:

- Neon provides built-in connection pooling
- Configure your application to use the pooled connection string (typically ends with `/dbname?sslmode=require&pool_timeout=30`)

## Free Tier Limitations

Be aware of the free tier limitations:

- **Compute**: 1 shared CPU, 1GB RAM
- **Storage**: 10GB of data
- **Databases**: Up to 3 databases per project
- **Connection Limits**: Limited concurrent connections
- **Activity Timeout**: Databases may pause after 5 minutes of inactivity

## Best Practices for Free Tier

1. **Connection Management**: Properly close database connections to avoid hitting connection limits
2. **Data Size**: Monitor your data size and implement cleanup strategies if needed
3. **Activity**: Be aware that databases may pause and need a moment to resume on first access
4. **Use Branches**: Leverage Neon's branching for development and testing

## Testing the Connection

You can test your Neon connection using the following Python code:

```python
import asyncpg
import asyncio

async def test_connection():
    conn = await asyncpg.connect(
        "postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require"
    )

    try:
        result = await conn.fetchval("SELECT version();")
        print(f"Connection successful! PostgreSQL version: {result}")
    finally:
        await conn.close()

# Run the test
asyncio.run(test_connection())
```

## Troubleshooting

- **Connection Issues**: Verify your connection string and credentials
- **SSL Errors**: Ensure you're using `sslmode=require` in your connection string
- **Paused Databases**: If you get timeout errors, try connecting again as the database may need to resume
- **Connection Limits**: Implement proper connection closing and consider connection pooling

## Next Steps

Once you have Neon set up, you can proceed to integrate it with your application by implementing the database service that will handle document storage and chat history management.