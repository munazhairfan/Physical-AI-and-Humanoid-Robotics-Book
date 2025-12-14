require('dotenv').config();
const { Pool } = require('pg');
const fs = require('fs');

const databaseUrl = process.env.DATABASE_URL;

const pool = new Pool({
  connectionString: databaseUrl,
  ssl: {
    rejectUnauthorized: false
  }
});

async function executeSchema() {
  try {
    console.log('Connecting to database to create Better Auth tables...');
    const client = await pool.connect();

    const sql = fs.readFileSync('create-better-auth-tables.sql', 'utf8');

    console.log('Executing schema creation...');
    await client.query(sql);
    console.log('Schema created successfully!');

    // Verify tables were created
    const result = await client.query(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      AND table_name LIKE 'auth_%'
    `);

    console.log('Created tables:', result.rows.map(row => row.table_name));

    client.release();
  } catch (error) {
    console.error('Error creating schema:', error.message);
  } finally {
    await pool.end();
  }
}

executeSchema();