# Data Model: Locale Switching Guard

## Key Entities

### Locale
- **Definition**: Represents a specific language/country variant (e.g., en, es, fr) with associated translations
- **Attributes**:
  - code: string (e.g., "en", "es", "fr")
  - name: string (e.g., "English", "Spanish", "French")
  - isDefault: boolean

### Documentation Path
- **Definition**: Represents the hierarchical path to a specific documentation page
- **Attributes**:
  - path: string (e.g., "/docs/intro", "/docs/api/reference")
  - localePrefix: string (e.g., "", "/es", "/fr")
  - fullPath: string (e.g., "/docs/intro", "/es/docs/intro")

### Page Translation Set
- **Definition**: Group of documentation pages that represent the same content in different locales
- **Attributes**:
  - basePath: string (e.g., "/docs/intro")
  - translations: Map<Locale, DocumentationPath>
  - availableLocales: Locale[]

## Relationships
- A Locale can have multiple Documentation Paths
- A Documentation Path belongs to one Locale
- A Page Translation Set contains multiple Documentation Paths across different Locales