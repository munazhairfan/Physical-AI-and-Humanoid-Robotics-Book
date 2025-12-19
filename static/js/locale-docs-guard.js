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
    // Check if path starts with /docs/ or /{locale}/docs/
    return pathname.startsWith('/docs/') ||
           /^\/[a-zA-Z]{2}\/docs\//.test(pathname) ||
           /^\/[a-zA-Z]{2}-[a-zA-Z]{2}\/docs\//.test(pathname);
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathParts = window.location.pathname.split('/');
    // Check if the first part after domain is a locale code (e.g., 'en', 'es', 'fr', 'ur')
    if (pathParts.length > 1 && pathParts[1] !== 'docs') {
      // It might be a locale prefix like /en/docs/, /es/docs/, /ur/docs/, etc.
      const potentialLocale = pathParts[1];
      // Basic check for locale format (2-3 letters)
      if (/^[a-zA-Z]{2,3}$/.test(potentialLocale)) {
        return potentialLocale;
      }
    }
    return null; // Default locale (no prefix)
  }

  // Generate the target locale-specific docs path
  function getTargetDocsPath(targetLocale) {
    const currentPath = window.location.pathname;
    const currentLocale = getCurrentLocale();

    if (currentLocale) {
      // Current path has locale prefix: /locale/docs/...
      // Remove the current locale prefix and add the target locale
      const pathWithoutLocale = currentPath.substring(currentLocale.length + 1); // +1 for the '/'
      if (targetLocale) {
        return '/' + targetLocale + pathWithoutLocale;
      } else {
        // Target is default locale (no prefix)
        return pathWithoutLocale;
      }
    } else {
      // Current path has no locale prefix (default locale)
      // Add the target locale prefix to the docs path
      if (targetLocale) {
        return '/' + targetLocale + currentPath;
      } else {
        // Already on default locale
        return currentPath;
      }
    }
  }

  // Function to handle locale switching on docs pages
  function handleDocsLocaleSwitch(targetLocale) {
    // Generate the correct docs path for the target locale
    const targetDocsPath = getTargetDocsPath(targetLocale);

    // Redirect to the locale-specific docs path
    window.location.href = targetDocsPath;
  }

  // Override the native history.pushState and location assignment to intercept locale changes
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;

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
      if (href && (href.startsWith('/en/') || href.startsWith('/es/') || href.startsWith('/fr/') ||
                   href.startsWith('/de/') || href.startsWith('/ja/') || href.startsWith('/ko/') ||
                   href.startsWith('/zh/') || href.startsWith('/ur/'))) {

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

      // Also check if the element has a locale-related class or is part of locale dropdown
      const className = targetElement.className || '';
      const textContent = targetElement.textContent || '';
      if (className.includes('locale') || className.includes('dropdown') ||
          textContent.toLowerCase().includes('locale') || textContent.toLowerCase().includes('language')) {
        // Wait a moment to see if a locale link is clicked within the dropdown
        setTimeout(() => {
          if (isDocsPage()) {
            // If still on a docs page after the locale switch, handle it
            const currentLocaleMatch = window.location.pathname.match(/^\/([a-zA-Z-]{2,5})\//);
            if (currentLocaleMatch) {
              const targetLocale = currentLocaleMatch[1];
              // Double check that we're navigating to a locale path
              if (window.location.pathname !== window.location.href) {
                handleDocsLocaleSwitch(targetLocale);
              }
            }
          }
        }, 100);
      }

      targetElement = targetElement.parentElement;
    }
  }, true);

  // Additionally, monitor for URL changes that might indicate locale switching
  let lastPathname = window.location.pathname;
  setInterval(() => {
    if (window.location.pathname !== lastPathname) {
      // Check if we just switched from a docs page to a different locale
      if (lastPathname.startsWith('/docs/') && window.location.pathname.startsWith('/docs/')) {
        // This means we're still on docs but might have switched locale without proper handling
        // Check if the new URL has a locale prefix where the old one didn't, or vice versa
        const isNewLocalePrefixed = /^\/[a-zA-Z-]{2,5}\/docs\//.test(window.location.pathname);
        const wasOldLocalePrefixed = /^\/[a-zA-Z-]{2,5}\/docs\//.test(lastPathname);

        if (isNewLocalePrefixed !== wasOldLocalePrefixed) {
          // Locale switch occurred, make sure we handle it properly
          if (isDocsPage()) {
            // If we're still on a docs page, ensure proper locale handling
            const currentLocaleMatch = window.location.pathname.match(/^\/([a-zA-Z-]{2,5})\//);
            if (currentLocaleMatch) {
              const currentLocale = currentLocaleMatch[1];
              // Redirect to ensure proper locale-specific docs path
              const targetDocsPath = getTargetDocsPath(currentLocale);
              if (window.location.pathname !== targetDocsPath) {
                window.location.replace(targetDocsPath);
              }
            }
          }
        }
      }
      lastPathname = window.location.pathname;
    }
  }, 500);

  // Also observe DOM changes for dynamically added locale switchers
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === 'childList') {
        mutation.addedNodes.forEach(function(node) {
          if (node.nodeType === 1) { // Element node
            // Check if this is a locale switcher
            const localeLinks = node.querySelectorAll?.('a[href^="/en/"], a[href^="/es/"], a[href^="/fr/"], a[href^="/de/"], a[href^="/ja/"], a[href^="/ko/"], a[href^="/zh/"], a[href^="/ur/"]') || [];
            localeLinks.forEach(function(link) {
              // The event listener above should handle these automatically
            });
          }
        });
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

})();