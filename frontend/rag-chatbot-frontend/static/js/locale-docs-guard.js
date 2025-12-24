/**
 * Locale Switching Guard for Docusaurus Docs Pages
 *
 * This script prevents 404 errors when switching locale from docs pages
 * by intercepting the locale switch event and redirecting to appropriate
 * locale-specific documentation paths.
 */

(function() {
  'use strict';

  // Check if we're on a docs page
  function isDocsPage() {
    const pathname = window.location.pathname;
    // Check if path starts with /docs/ or /{locale}/docs/ or contains docs after locale
    return pathname.startsWith('/docs/') ||
           pathname.includes('/docs/') ||
           /^\/[a-zA-Z]{2}\/docs\//.test(pathname) ||
           /^\/[a-zA-Z]{2}-[a-zA-Z]{2}\/docs\//.test(pathname);
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathname = window.location.pathname;
    const pathParts = pathname.split('/').filter(part => part !== '');

    // Check if the first part is a locale code (e.g., 'en', 'es', 'fr', 'ur')
    if (pathParts.length > 0) {
      const potentialLocale = pathParts[0];

      // Check if this is a locale code pattern (2-3 letters)
      if (/^[a-zA-Z]{2,3}$/.test(potentialLocale)) {
        // Verify that the next part is 'docs' to confirm it's a locale-prefixed docs path
        if (pathParts.length > 1 && pathParts[1] === 'docs') {
          return potentialLocale;
        }
      }
    }
    return null; // Default locale (no prefix)
  }

  // Generate the target locale-specific docs path
  function getTargetDocsPath(targetLocale) {
    const currentPath = window.location.pathname;
    const currentLocale = getCurrentLocale();

    let docsPath;

    // Extract the docs part of the path regardless of locale prefix
    if (currentLocale) {
      // Path has locale prefix like /ur/docs/intro, extract /docs/intro part
      docsPath = currentPath.substring(currentLocale.length + 1); // +1 for the '/'
    } else {
      // Path is in default format like /docs/intro or /intro
      docsPath = currentPath;
    }

    // Ensure the path starts with /docs/ if it doesn't already
    if (!docsPath.startsWith('/docs/')) {
      // If it starts with /intro, /chapter1, etc., prepend /docs/
      if (docsPath.startsWith('/')) {
        docsPath = '/docs' + docsPath;
      } else {
        docsPath = '/docs/' + docsPath;
      }
    }

    // Add the target locale prefix if needed (not for default locale)
    if (targetLocale && targetLocale !== 'en') { // Assuming 'en' is the default locale
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
    window.location.href = targetDocsPath;
  }

  // Listen for clicks on locale switcher elements
  document.addEventListener('click', function(event) {
    // Only handle if we're on a docs page
    if (!isDocsPage()) {
      return;
    }

    // Check if the clicked element is a locale switcher
    let targetElement = event.target;
    while (targetElement && targetElement !== document) {
      const href = targetElement.getAttribute?.('href');
      if (href && /^[a-zA-Z]{2,3}(-[a-zA-Z]{2,3})?\//.test(href.split('/')[1])) {

        // Prevent the default behavior
        event.preventDefault();

        // Extract the target locale from the href
        const localeMatch = href.match(/^\/([a-zA-Z-]{2,5})\//);
        if (localeMatch) {
          const targetLocale = localeMatch[1];
          handleDocsLocaleSwitch(targetLocale);
        }
        break;
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

    // Check if this navigation is happening on a docs page and involves locale change
    const newUrl = arguments[2];
    if (newUrl && typeof newUrl === 'string' && isDocsPage()) {
      const localeMatch = newUrl.match(/^\/([a-zA-Z]{2,3}(-[a-zA-Z]{2,3})?)\//);
      if (localeMatch) {
        const targetLocale = localeMatch[1];
        // If we're on a docs page and changing locale, ensure proper path
        setTimeout(() => {
          if (isDocsPage()) {
            const correctPath = getTargetDocsPath(targetLocale);
            if (window.location.pathname !== correctPath) {
              window.location.replace(correctPath);
            }
          }
        }, 0);
      }
    }
  };

  history.replaceState = function() {
    originalReplaceState.apply(history, arguments);

    // Similar check for replaceState
    const newUrl = arguments[2];
    if (newUrl && typeof newUrl === 'string' && isDocsPage()) {
      const localeMatch = newUrl.match(/^\/([a-zA-Z]{2,3}(-[a-zA-Z]{2,3})?)\//);
      if (localeMatch) {
        const targetLocale = localeMatch[1];
        setTimeout(() => {
          if (isDocsPage()) {
            const correctPath = getTargetDocsPath(targetLocale);
            if (window.location.pathname !== correctPath) {
              window.location.replace(correctPath);
            }
          }
        }, 0);
      }
    }
  };

  // Listen for popstate events (browser back/forward buttons)
  window.addEventListener('popstate', function(event) {
    if (isDocsPage()) {
      // After navigation, check if we're on the right locale-specific path
      const currentLocale = getCurrentLocale();
      if (currentLocale) {
        const expectedPath = getTargetDocsPath(currentLocale);
        if (window.location.pathname !== expectedPath && window.location.pathname.startsWith('/docs/')) {
          // Redirect to the proper locale-specific docs path
          window.location.replace(expectedPath);
        }
      }
    }
  });

})();