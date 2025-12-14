require('dotenv').config();
const { Pool } = require('pg');

const databaseUrl = process.env.DATABASE_URL;

const pool = new Pool({
  connectionString: databaseUrl,
  ssl: {
    rejectUnauthorized: false
  }
});

async function checkSchema() {
  try {
    console.log('Connecting to database to check schema...');
    const client = await pool.connect();

    // Check if Better Auth tables exist
    const tables = ['auth_user', 'auth_session', 'auth_account', 'auth_verification_token'];

    for (const table of tables) {
      try {
        const result = await client.query(`SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_name = '${table}'
        );`);
        console.log(`Table ${table} exists:`, result.rows[0].exists);
      } catch (error) {
        console.log(`Table ${table} exists: false (error: ${error.message})`);
      }
    }

    // Check for common Better Auth table names
    const commonTables = ['user', 'session', 'account', 'verification_token'];
    for (const table of commonTables) {
      try {
        const result = await client.query(`SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_name = '${table}'
        );`);
        console.log(`Table ${table} exists:`, result.rows[0].exists);
      } catch (error) {
        console.log(`Table ${table} exists: false (error: ${error.message})`);
      }
    }

    // List all tables in the database
    const allTables = await client.query(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
    `);
    console.log('All tables in database:', allTables.rows.map(row => row.table_name));

    client.release();
  } catch (error) {
    console.error('Error checking schema:', error.message);
  } finally {
    await pool.end();
  }
}

checkSchema();