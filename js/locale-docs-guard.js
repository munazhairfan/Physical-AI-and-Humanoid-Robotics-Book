/**
 * Locale Switching Guard for Docusaurus Docs Pages
 *
 * This script prevents 404 errors when switching locale from docs pages
 * by intercepting the locale switch event and redirecting to appropriate
 * locale-specific documentation paths.
 */

(function() {
  'use strict';

  // Define supported locales
  const SUPPORTED_LOCALES = ['en', 'ur'];
  const DEFAULT_LOCALE = 'en';

  // Check if we're on a docs page
  function isDocsPage() {
    const pathname = window.location.pathname;
    // Check if path starts with /docs/ or contains docs after locale
    return pathname.startsWith('/docs/') ||
           pathname.includes('/docs/') ||
           SUPPORTED_LOCALES.some(locale => pathname.startsWith(`/${locale}/docs/`));
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathname = window.location.pathname;
    const pathParts = pathname.split('/').filter(part => part !== '');

    // Check if the first part is a supported locale code
    if (pathParts.length > 0 && SUPPORTED_LOCALES.includes(pathParts[0])) {
      return pathParts[0];
    }
    return DEFAULT_LOCALE; // Default locale (no prefix)
  }

  // Get the current docs path without locale prefix
  function getCurrentDocsPath() {
    const pathname = window.location.pathname;
    const currentLocale = getCurrentLocale();

    if (currentLocale !== DEFAULT_LOCALE) {
      // Path has locale prefix like /ur/docs/intro, extract /docs/intro part
      return pathname.substring(currentLocale.length + 1); // +1 for the '/'
    } else {
      // Path is in default format like /docs/intro or /intro
      if (pathname.startsWith('/docs/')) {
        return pathname;
      } else if (pathname !== '/' && !pathname.startsWith('/docs/')) {
        // If it's a specific page but not in docs, assume it's a doc page
        return '/docs' + pathname;
      }
      return pathname;
    }
  }

  // Generate the target locale-specific docs path
  function getTargetDocsPath(targetLocale) {
    let docsPath = getCurrentDocsPath();

    // Ensure the path starts with /docs/
    if (!docsPath.startsWith('/docs/')) {
      if (docsPath.startsWith('/')) {
        docsPath = '/docs' + docsPath;
      } else {
        docsPath = '/docs/' + docsPath;
      }
    }

    // Add the target locale prefix if needed (not for default locale)
    if (targetLocale && targetLocale !== DEFAULT_LOCALE) {
      return '/' + targetLocale + docsPath;
    } else {
      // Return docs path without locale prefix (for default locale)
      return docsPath;
    }
  }

  // Function to handle locale switching on docs pages
  function handleDocsLocaleSwitch(targetLocale) {
    // Generate the correct docs path for the target locale
    const targetDocsPath = getTargetDocsPath(targetLocale);

    // Redirect to the locale-specific docs path
    window.location.href = targetDocsPath + window.location.search + window.location.hash;
  }

  // Listen for clicks on locale switcher elements
  document.addEventListener('click', function(event) {
    // Check if the clicked element is a locale switcher
    let targetElement = event.target;
    while (targetElement && targetElement !== document) {
      // Check if this is a locale dropdown item (typically has data-locale attribute or href with locale)
      const href = targetElement.getAttribute?.('href');
      if (href) {
        const localeMatch = href.match(/^\/([a-zA-Z-]{2,5})(\/.*)?$/);
        if (localeMatch) {
          const targetLocale = localeMatch[1];

          // Only handle if it's a supported locale
          if (SUPPORTED_LOCALES.includes(targetLocale)) {
            // Check if we're on a docs page
            if (isDocsPage()) {
              // Prevent the default behavior
              event.preventDefault();
              handleDocsLocaleSwitch(targetLocale);
              return;
            }
          }
        }
      }

      targetElement = targetElement.parentElement;
    }
  }, true);

  // Docusaurus may use history.pushState for client-side navigation
  // Let's intercept that as well
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;

  // Override pushState to detect locale changes
  history.pushState = function() {
    originalPushState.apply(history, arguments);
  };

  history.replaceState = function() {
    originalReplaceState.apply(history, arguments);
  };

  // Listen for popstate events (browser back/forward buttons)
  window.addEventListener('popstate', function(event) {
    // After navigation, check if we're on the right locale-specific path
    if (isDocsPage()) {
      const currentLocale = getCurrentLocale();
      if (currentLocale) {
        const expectedPath = getTargetDocsPath(currentLocale);
        if (window.location.pathname !== expectedPath && window.location.pathname.startsWith('/docs/')) {
          // Redirect to the proper locale-specific docs path
          window.location.replace(expectedPath + window.location.search + window.location.hash);
        }
      }
    }
  });

})();